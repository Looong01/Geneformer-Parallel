import logging
import math
import pickle
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from sklearn import preprocessing
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from tqdm.auto import trange

from . import TOKEN_DICTIONARY_FILE
from .emb_extractor import make_colorbar


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


def preprocess_classifier_batch(cell_batch, max_len, label_name):
    if max_len is None:
        max_len = max([len(i) for i in cell_batch["input_ids"]])

    # load token dictionary (Ensembl IDs:token)
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        gene_token_dict = pickle.load(f)

    def pad_label_example(example):
        example[label_name] = np.pad(
            example[label_name],
            (0, max_len - len(example["input_ids"])),
            mode="constant",
            constant_values=-100,
        )
        example["input_ids"] = np.pad(
            example["input_ids"],
            (0, max_len - len(example["input_ids"])),
            mode="constant",
            constant_values=gene_token_dict.get("<pad>"),
        )
        example["attention_mask"] = (
            example["input_ids"] != gene_token_dict.get("<pad>")
        ).astype(int)
        return example

    padded_batch = cell_batch.map(pad_label_example)
    return padded_batch


# Function to find the largest number smaller
# than or equal to N that is divisible by k
def find_largest_div(N, K):
    rem = N % K
    if rem == 0:
        return N
    else:
        return N - rem


def vote(logit_list):
    m = max(logit_list)
    logit_list.index(m)
    indices = [i for i, x in enumerate(logit_list) if x == m]
    if len(indices) > 1:
        return "tie"
    else:
        return indices[0]


def py_softmax(vector):
    e = np.exp(vector)
    return e / e.sum()


def classifier_predict(model, classifier_type, evalset, forward_batch_size):
    if classifier_type == "gene":
        label_name = "labels"
    elif classifier_type == "cell":
        label_name = "label"

    predict_logits = []
    predict_labels = []
    model.eval()
    if isinstance(model, torch.nn.DataParallel):
        device_ids = model.device_ids  # 获取多GPU设备列表
        main_device_id = random.choice(device_ids)  # 随机选择一个GPU设备ID
        main_device = f"cuda:{main_device_id}"  # 主设备设为随机选择的设备
    else:
        main_device = "cuda"  # 单GPU

    # ensure there is at least 2 examples in each batch to avoid incorrect tensor dims
    evalset_len = len(evalset)
    max_divisible = find_largest_div(evalset_len, forward_batch_size)
    if len(evalset) - max_divisible == 1:
        evalset_len = max_divisible

    max_evalset_len = max(evalset.select([i for i in range(evalset_len)])["length"])

    disable_progress_bar()  # disable progress bar for preprocess_classifier_batch mapping
    for i in trange(0, evalset_len, forward_batch_size):
        max_range = min(i + forward_batch_size, evalset_len)
        batch_evalset = evalset.select([i for i in range(i, max_range)])
        padded_batch = preprocess_classifier_batch(
            batch_evalset, max_evalset_len, label_name
        )
        padded_batch.set_format(type="torch")

        input_data_batch = padded_batch["input_ids"]
        attn_msk_batch = padded_batch["attention_mask"]
        label_batch = padded_batch[label_name]
        
        # 数据移动到主GPU，DataParallel会自动分发到其他GPU
        input_data_batch = input_data_batch.to(main_device)
        attn_msk_batch = attn_msk_batch.to(main_device)
        label_batch = label_batch.to(main_device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_data_batch,
                attention_mask=attn_msk_batch,
                labels=label_batch,
            )
            predict_logits += [torch.squeeze(outputs.logits.to("cpu"))]
            predict_labels += [torch.squeeze(label_batch.to("cpu"))]

    enable_progress_bar()
    logits_by_cell = torch.cat(predict_logits)
    last_dim = len(logits_by_cell.shape) - 1
    all_logits = logits_by_cell.reshape(-1, logits_by_cell.shape[last_dim])
    labels_by_cell = torch.cat(predict_labels)
    all_labels = torch.flatten(labels_by_cell)
    logit_label_paired = [
        item
        for item in list(zip(all_logits.tolist(), all_labels.tolist()))
        if item[1] != -100
    ]
    y_pred = [vote(item[0]) for item in logit_label_paired]
    y_true = [item[1] for item in logit_label_paired]
    logits_list = [item[0] for item in logit_label_paired]
    return y_pred, y_true, logits_list


def get_metrics(y_pred, y_true, logits_list, num_classes, labels):
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(labels))
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    roc_metrics = None  # roc metrics not reported for multiclass
    if num_classes == 2:
        y_score = [py_softmax(item)[1] for item in logits_list]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        mean_fpr = np.linspace(0, 1, 100)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_wt = len(tpr)
        roc_auc = auc(fpr, tpr)
        roc_metrics = {
            "fpr": fpr,
            "tpr": tpr,
            "interp_tpr": interp_tpr,
            "auc": roc_auc,
            "tpr_wt": tpr_wt,
        }
    return conf_mat, macro_f1, acc, roc_metrics


# get cross-validated mean and sd metrics
def get_cross_valid_roc_metrics(all_tpr, all_roc_auc, all_tpr_wt):
    wts = [count / sum(all_tpr_wt) for count in all_tpr_wt]
    all_weighted_tpr = [a * b for a, b in zip(all_tpr, wts)]
    mean_tpr = np.sum(all_weighted_tpr, axis=0)
    mean_tpr[-1] = 1.0
    all_weighted_roc_auc = [a * b for a, b in zip(all_roc_auc, wts)]
    roc_auc = np.sum(all_weighted_roc_auc)
    roc_auc_sd = math.sqrt(np.average((all_roc_auc - roc_auc) ** 2, weights=wts))
    return mean_tpr, roc_auc, roc_auc_sd


# plot ROC curve
def plot_ROC(roc_metric_dict, model_style_dict, title, output_dir, output_prefix):
    fig = plt.figure()
    fig.set_size_inches(10, 8)
    sns.set(font_scale=2)
    sns.set_style("white")
    lw = 3
    for model_name in roc_metric_dict.keys():
        mean_fpr = roc_metric_dict[model_name]["mean_fpr"]
        mean_tpr = roc_metric_dict[model_name]["mean_tpr"]
        color = model_style_dict[model_name]["color"]
        linestyle = model_style_dict[model_name]["linestyle"]
        if "roc_auc" not in roc_metric_dict[model_name].keys():
            all_roc_auc = roc_metric_dict[model_name]["all_roc_auc"]
            label = f"{model_name} (AUC {all_roc_auc:0.2f})"
        else:
            roc_auc = roc_metric_dict[model_name]["roc_auc"]
            roc_auc_sd = roc_metric_dict[model_name]["roc_auc_sd"]
            if len(roc_metric_dict[model_name]["all_roc_auc"]) > 1:
                label = f"{model_name} (AUC {roc_auc:0.2f} $\pm$ {roc_auc_sd:0.2f})"
            else:
                label = f"{model_name} (AUC {roc_auc:0.2f})"
        plt.plot(
            mean_fpr, mean_tpr, color=color, linestyle=linestyle, lw=lw, label=label
        )

    plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    output_file = (Path(output_dir) / f"{output_prefix}_roc").with_suffix(".pdf")
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()


# plot confusion matrix
def plot_confusion_matrix(
    conf_mat_df, title, output_dir, output_prefix, custom_class_order
):
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {"axes.grid": False})
    if custom_class_order is not None:
        conf_mat_df = conf_mat_df.reindex(
            index=custom_class_order, columns=custom_class_order
        )
    display_labels = generate_display_labels(conf_mat_df)
    conf_mat = preprocessing.normalize(conf_mat_df.to_numpy(), norm="l1")
    display = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=display_labels
    )
    display.plot(cmap="Blues", values_format=".2g")
    plt.title(title)
    plt.show()

    output_file = (Path(output_dir) / f"{output_prefix}_conf_mat").with_suffix(".pdf")
    display.figure_.savefig(output_file, bbox_inches="tight")


def generate_display_labels(conf_mat_df):
    display_labels = []
    i = 0
    for label in conf_mat_df.index:
        display_labels += [f"{label}\nn={conf_mat_df.iloc[i,:].sum():.0f}"]
        i = i + 1
    return display_labels


def plot_predictions(predictions_df, title, output_dir, output_prefix, kwargs_dict):
    sns.set(font_scale=2)
    plt.figure(figsize=(10, 10), dpi=150)
    label_colors, label_color_dict = make_colorbar(predictions_df, "true")
    predictions_df = predictions_df.drop(columns=["true"])
    predict_colors_list = [label_color_dict[label] for label in predictions_df.columns]
    predict_label_list = [label for label in predictions_df.columns]
    predict_colors = pd.DataFrame(
        pd.Series(predict_colors_list, index=predict_label_list), columns=["predicted"]
    )

    default_kwargs_dict = {
        "row_cluster": False,
        "col_cluster": False,
        "row_colors": label_colors,
        "col_colors": predict_colors,
        "linewidths": 0,
        "xticklabels": False,
        "yticklabels": False,
        "center": 0,
        "cmap": "vlag",
    }

    if kwargs_dict is not None:
        default_kwargs_dict.update(kwargs_dict)
    g = sns.clustermap(predictions_df, **default_kwargs_dict)

    plt.setp(g.ax_row_colors.get_xmajorticklabels(), rotation=45, ha="right")

    for label_color in list(label_color_dict.keys()):
        g.ax_col_dendrogram.bar(
            0, 0, color=label_color_dict[label_color], label=label_color, linewidth=0
        )

        g.ax_col_dendrogram.legend(
            title=f"{title}",
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, 1),
            facecolor="white",
        )

    output_file = (Path(output_dir) / f"{output_prefix}_pred").with_suffix(".pdf")
    plt.savefig(output_file, bbox_inches="tight")


# def classifier_predict_DDP(model, classifier_type, evalset, forward_batch_size, local_rank):
#     # 初始化分布式环境（需在外部调用 dist.init_process_group 和设置 local_rank）
#     device = torch.device(f"cuda:{local_rank}")
#     torch.cuda.set_device(device)
    
#     if classifier_type == "gene":
#         label_name = "labels"
#     elif classifier_type == "cell":
#         label_name = "label"

#     # 包装模型为 DDP
#     model = DDP(model.to(device), device_ids=[local_rank], output_device=local_rank)
#     model.eval()
    
#     # ensure there is at least 2 examples in each batch to avoid incorrect tensor dims
#     evalset_len = len(evalset)
#     max_divisible = find_largest_div(evalset_len, forward_batch_size)
#     if len(evalset) - max_divisible == 1:
#         evalset_len = max_divisible

#     max_evalset_len = max(evalset.select([i for i in range(evalset_len)])["length"])
    
#     # 创建分布式采样器
#     sampler = DistributedSampler(
#         evalset,
#         num_replicas=dist.get_world_size(),
#         rank=dist.get_rank(),
#         shuffle=False
#     )
    
#     # 调整批次大小确保可被分布式分片整除
#     forward_batch_size = adjust_batch_size(forward_batch_size, len(evalset), dist.get_world_size())
    
#     # 创建 DataLoader
#     batch_evalset = preprocess_classifier_batch(
#         evalset, max_evalset_len, label_name
#     )
#     eval_loader = DataLoader(
#         evalset,
#         batch_size=forward_batch_size,
#         sampler=sampler,
#         # collate_fn=batch_evalset,
#         pin_memory=True
#     )

#     predict_logits = []
#     predict_labels = []
    
#     # 只在主进程显示进度条
#     if dist.get_rank() == 0:
#         eval_iterator = trange(len(eval_loader), desc="Predicting")
#     else:
#         eval_iterator = range(len(eval_loader))
    
#     disable_progress_bar()  # 禁用底层进度条
    
#     for batch in eval_loader:
#         # 将数据移动到当前设备
#         input_ids = batch["input_ids"].to(device, non_blocking=True)
#         attention_mask = batch["attention_mask"].to(device, non_blocking=True)
#         labels = batch[label_name].to(device, non_blocking=True)
        
#         with torch.no_grad():
#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels
#             )
            
#         # 收集当前进程的结果
#         predict_logits.append(outputs.logits.detach().cpu())
#         predict_labels.append(labels.detach().cpu())

#     # 汇总所有进程的结果
#     all_logits = [torch.zeros_like(predict_logits[0]) for _ in range(dist.get_world_size())]
#     all_labels = [torch.zeros_like(predict_labels[0]) for _ in range(dist.get_world_size())]
    
#     dist.all_gather(all_logits, predict_logits[0])
#     dist.all_gather(all_labels, predict_labels[0])
    
#     # 只在主进程处理最终结果
#     if dist.get_rank() == 0:
#         all_logits = torch.cat(all_logits)
#         all_labels = torch.cat(all_labels)
        
#         logit_label_paired = [
#             item for item in zip(all_logits.tolist(), all_labels.tolist()) 
#             if item[1] != -100
#         ]
#         y_pred = [vote(item[0]) for item in logit_label_paired]
#         y_true = [item[1] for item in logit_label_paired]
#         logits_list = [item[0] for item in logit_label_paired]
#         return y_pred, y_true, logits_list
#     else:
#         return None, None, None

# def adjust_batch_size(batch_size, dataset_size, num_gpus):
#     world_size = num_gpus
#     if batch_size % world_size != 0:
#         batch_size = batch_size // world_size * world_size
#     if dataset_size % batch_size == 1:  # 避免最后一个批次只有1个样本
#         batch_size -= 1
#     return batch_size