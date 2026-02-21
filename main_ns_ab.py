import sys

sys.path.append("../")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import numpy as np

from NS_Merging.task_vectors import TaskVector
from NS_Merging.eval import eval_single_dataset
from NS_Merging.args import parse_arguments
from NS_Merging.ns_merging_utils import ns_merging

torch.set_num_interop_threads(1)


def setup_logger(log_dir, filename="log.txt"):
    import logging

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(f"{log_dir}/{filename}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


merge_datasets = [
    "SVHN",
    "GTSRB",
    "EuroSAT",
    "MNIST",
    "DTD",
    "RESISC45",
    "Cars",
    "SUN397",
]


eval_datasets = [
    "MNIST",
    "EuroSAT",
    "GTSRB",
    "SVHN",
    "RESISC45",
    "DTD",
    "Cars",
    "SUN397",
]

args = parse_arguments()
args.repeat = 1  # 实验次数
args.scaling_coef_ = 0.8  # \alpha

args.ratio = 1  # 线性层组件开关：0 => 关闭NS投影；1 => 开启NS投影
args.log_ns_cosine = False  # 是否计算任务向量间的 COS ，按照 merge_datasets 的顺序

if not hasattr(args, "ties_reset_thresh_ln"):
    args.ties_reset_thresh_ln = 0.4
if not hasattr(args, "ties_reset_thresh_bias"):
    args.ties_reset_thresh_bias = 0.4

model_name = "ViT-B-32"
args.exp_size = 1  # 样本数，由于VIT32B参数量较小，所以一个最优，否则容易限制过死
args.data_location = "./data"
args.model = model_name
args.device = "cuda"
args.save = "./checkpoints/" + model_name
args.logs_path = "./logs/" + model_name
base_checkpoint_path = "./checkpoints/" + model_name + "/zeroshot.pt"

ablation_settings = [
    ("full", 1, 0.4, 0.4),  # NS on, Ties on
    ("w_o_ns", 0, 0.4, 0.4),  # NS off, Ties on
    ("w_o_ties", 1, 0.0, 0.0),  # NS on, Ties off
    ("w_o_both", 0, 0.0, 0.0),  # NS off, Ties off
]

timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

for merge_config_id in [0]:
    for tag, ratio, ties_ln, ties_bias in ablation_settings:
        args.ratio = ratio
        args.ties_reset_thresh_ln = ties_ln
        args.ties_reset_thresh_bias = ties_bias

        logger = setup_logger(args.logs_path, f"log_{timestamp}_{tag}.txt")

        print("################################################################")
        print(
            "######################### Merging :",
            merge_config_id,
            f" | Ablation : {tag}",
            " ##############################",
        )
        print("################################################################")
        print(args)
        logger.info(f"========== Ablation: {tag} ==========")
        logger.info(str(args))

        all_run_accuracies = []
        for run_idx in range(args.repeat):
            print(
                "######################### Run :",
                run_idx,
                f" | Ablation : {tag}",
                " ##############################",
            )
            logger.info(f"========== Run {run_idx} ({tag}) ==========")

            task_vectors = [
                TaskVector(
                    base_checkpoint_path,
                    "./checkpoints/"
                    + model_name
                    + "/"
                    + dataset_name
                    + "/finetuned.pt",
                )
                for dataset_name in merge_datasets
            ]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            task_vectors = ns_merging(
                args, task_vectors, base_checkpoint_path, merge_datasets
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            logger.info(f"[TIME] ns_merging total: {end_time - start_time:.2f} s")
            print(f"[TIME] ns_merging total: {end_time - start_time:.2f} s")

            merged_task_vector = sum(task_vectors)
            merged_image_encoder = merged_task_vector.apply_to(  # pyright: ignore[reportAttributeAccessIssue]
                base_checkpoint_path, scaling_coef=args.scaling_coef_
            )
            logger.info("*" * 20 + "scaling_coef:" + str(args.scaling_coef_) + "*" * 20)

            dataset_top1_accs = []
            for dataset_name in eval_datasets:
                eval_metrics = eval_single_dataset(
                    merged_image_encoder, dataset_name, args
                )
                acc = eval_metrics.get("top1", 0.0) * 100
                logger.info(f"{dataset_name}:{acc}%")
                dataset_top1_accs.append(acc)

            logger.info("Avg ACC:" + str(np.mean(dataset_top1_accs)) + "%")
            all_run_accuracies.append(dataset_top1_accs)

        # 汇总统计
        all_run_accuracies = np.array(all_run_accuracies)

        per_dataset_mean = np.mean(all_run_accuracies, axis=0)
        per_dataset_std = np.std(all_run_accuracies, axis=0)

        logger.info("\n############# ACC for each dataset #############")
        for dataset_idx, dataset_name in enumerate(eval_datasets):
            logger.info(
                f"{dataset_name}: AVG={per_dataset_mean[dataset_idx]:.2f}%, STD={per_dataset_std[dataset_idx]:.2f}%"
            )

        per_run_mean = np.mean(all_run_accuracies, axis=1)
        overall_std = np.std(per_run_mean)
        overall_mean = np.mean(per_run_mean)

        logger.info("\n##########################")
        logger.info(
            f"Average performance across all datasets: AVG={overall_mean:.2f}%, STD={overall_std:.2f}%"
        )
