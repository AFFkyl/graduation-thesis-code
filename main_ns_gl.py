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

    logger = logging.getLogger(log_dir)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_dir + "/" + filename)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# ===== Seen datasets: used for merging =====
merge_datasets = [
    "SVHN",
    "GTSRB",
    "DTD",
    "RESISC45",
    "Cars",
    "SUN397",
]

# ===== Evaluation split =====
eval_datasets_seen = [
    "SVHN",
    "GTSRB",
    "DTD",
    "RESISC45",
    "Cars",
    "SUN397",
]

eval_datasets_unseen = [
    "MNIST",
    "EuroSAT",
]

args = parse_arguments()
args.repeat = 1  # 实验次数
args.scaling_coef_ = 0.8  # \alpha
args.ratio = 1  # 线性层组件开关
args.log_ns_cosine = False  # 是否计算任务向量间的 COS ，按照 merge_datasets 的顺序

model_name = "ViT-B-32"
args.exp_size = 1
args.data_location = "./data"
args.model = model_name
args.device = "cuda"
args.save = "./checkpoints/" + model_name
args.logs_path = "./logs/" + model_name
base_checkpoint_path = "./checkpoints/" + model_name + "/zeroshot.pt"

timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
logger = setup_logger(args.logs_path, f"log_{timestamp}_generalization.txt")

for merge_config_id in [0]:
    print("################################################################")
    print(
        "######################### Merging :",
        merge_config_id,
        " ##############################",
    )
    print("################################################################")
    print(args)

    run_idx = 0
    print(
        "######################### Run :",
        run_idx,
        " ##############################",
    )

    # 1) Build task vectors for seen datasets (used for merging)
    task_vectors = [
        TaskVector(
            base_checkpoint_path,
            "./checkpoints/" + model_name + "/" + dataset_name + "/finetuned.pt",
        )
        for dataset_name in merge_datasets
    ]

    # 2) NS-merging
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    task_vectors = ns_merging(args, task_vectors, base_checkpoint_path, merge_datasets)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    logger.info(f"[TIME] ns_merging total: {end_time - start_time:.2f} s")
    print(f"[TIME] ns_merging total: {end_time - start_time:.2f} s")

    # 3) Apply merged task vector to base model
    merged_task_vector = sum(task_vectors)
    merged_image_encoder = merged_task_vector.apply_to(  # pyright: ignore[reportAttributeAccessIssue]
        base_checkpoint_path, scaling_coef=args.scaling_coef_
    )
    logger.info("*" * 20 + "scaling_coef:" + str(args.scaling_coef_) + "*" * 20)

    # 4) Evaluate on seen datasets
    logger.info("\n############# SEEN DATASETS (used in merging) #############")
    seen_accs = []
    for dataset_name in eval_datasets_seen:
        eval_metrics = eval_single_dataset(merged_image_encoder, dataset_name, args)
        acc = eval_metrics.get("top1", 0.0) * 100
        logger.info(f"{dataset_name}: {acc:.2f}%")
        seen_accs.append(acc)
    logger.info(f"Seen Avg ACC: {np.mean(seen_accs):.2f}%")

    # 5) Evaluate on unseen datasets (generalization)
    logger.info("\n############# UNSEEN DATASETS (generalization) #############")
    unseen_accs = []
    for dataset_name in eval_datasets_unseen:
        eval_metrics = eval_single_dataset(merged_image_encoder, dataset_name, args)
        acc = eval_metrics.get("top1", 0.0) * 100
        logger.info(f"{dataset_name}: {acc:.2f}%")
        unseen_accs.append(acc)
    logger.info(f"Unseen Avg ACC: {np.mean(unseen_accs):.2f}%")
