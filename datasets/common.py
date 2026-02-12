import os
import torch
import glob
import collections
import random
from tqdm import tqdm
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Sampler


# 关键：不要用 0。自动取一个相对稳妥的 worker 数
NUM_WORKERS = max(2, min(8, (os.cpu_count() or 4) - 1))


def _maybe_upgrade_dataloader(loader, device):
    if not isinstance(loader, DataLoader):
        return loader

    # 已经是多进程就不动
    if getattr(loader, "num_workers", 0) > 0:
        return loader

    # 若配置为 0，则不升级
    if NUM_WORKERS <= 0:
        return loader

    try:
        pin_memory = torch.device(device).type == "cuda"

        kwargs = dict(
            dataset=loader.dataset,
            batch_sampler=loader.batch_sampler,  # 最大程度保持 sampler/shuffle/drop_last 行为
            num_workers=NUM_WORKERS,
            collate_fn=loader.collate_fn,
            pin_memory=pin_memory,
        )

        # 注意：persistent_workers/prefetch_factor 只在 num_workers > 0 时合法
        if NUM_WORKERS > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4

        return DataLoader(**kwargs)

    except Exception as e:
        print(f"[Warning] Could not set num_workers={NUM_WORKERS}, fallback. Reason: {e}")
        return loader


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, path, transform, flip_label_prob=0.0):
        super().__init__(path, transform)
        self.flip_label_prob = flip_label_prob
        if self.flip_label_prob > 0:
            print(f"Flipping labels with probability {self.flip_label_prob}")
            num_classes = len(self.classes)
            for i in range(len(self.samples)):
                if random.random() < self.flip_label_prob:
                    new_label = random.randint(0, num_classes - 1)
                    self.samples[i] = (self.samples[i][0], new_label)

    def __getitem__(self, index):  # pyright: ignore[reportIncompatibleMethodOverride]
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {"images": image, "labels": label, "image_paths": self.samples[index][0]}


def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {"images": batch[0], "labels": batch[1]}
    elif len(batch) == 3:
        batch = {"images": batch[0], "labels": batch[1], "metadata": batch[2]}
    else:
        raise ValueError(f"Unexpected number of elements: {len(batch)}")

    return batch


def get_features_helper(image_encoder, dataloader, device):
    all_data = collections.defaultdict(list)

    device = torch.device(device)
    image_encoder = image_encoder.to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        image_encoder = torch.nn.DataParallel(
            image_encoder, device_ids=[x for x in range(torch.cuda.device_count())]
        )

    image_encoder.eval()

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)

            images = batch["images"]
            if torch.is_tensor(images):
                images = images.to(device, non_blocking=(device.type == "cuda"))

            features = image_encoder(images)

            all_data["features"].append(features.detach().cpu())

            for key, val in batch.items():
                if key == "images":
                    continue
                if hasattr(val, "cpu"):
                    val = val.cpu()
                    all_data[key].append(val)
                else:
                    all_data[key].extend(val)

    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()

    return all_data


def get_features(is_train, image_encoder, dataset, device):
    split = "train" if is_train else "val"
    dname = type(dataset).__name__

    cache_dir = None
    cached_files = []

    if image_encoder.cache_dir is not None:
        cache_dir = f"{image_encoder.cache_dir}/{dname}/{split}"
        cached_files = glob.glob(f"{cache_dir}/*")

    if image_encoder.cache_dir is not None and len(cached_files) > 0:
        print(f"Getting features from {cache_dir}")
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            data[name] = torch.load(cached_file, map_location="cpu")
    else:
        if cache_dir is None:
            print("Did not find cached features (cache_dir is None). Building from scratch.")
        else:
            print(f"Did not find cached features at {cache_dir}. Building from scratch.")

        loader = dataset.train_loader if is_train else dataset.test_loader

        # 把已有 loader 尽量升级到 num_workers（失败自动回退）
        loader = _maybe_upgrade_dataloader(loader, device)

        data = get_features_helper(image_encoder, loader, device)

        if image_encoder.cache_dir is None:
            print("Not caching because no cache directory was passed.")
        else:
            os.makedirs(cache_dir, exist_ok=True)  # pyright: ignore[reportArgumentType]
            print(f"Caching data at {cache_dir}")
            for name, val in data.items():
                torch.save(val, f"{cache_dir}/{name}.pt")
    return data


class FeatureDataset(Dataset):
    def __init__(self, is_train, image_encoder, dataset, device):
        self.data = get_features(is_train, image_encoder, dataset, device)

    def __len__(self):
        return len(self.data["features"])

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.data.items()}
        data["features"] = torch.from_numpy(data["features"]).float()
        return data


def get_dataloader(dataset, is_train, args, image_encoder=None):
    if image_encoder is not None:
        feature_dataset = FeatureDataset(is_train, image_encoder, dataset, args.device)
        pin_memory = torch.device(args.device).type == "cuda"

        kwargs = dict(
            batch_size=args.batch_size,
            shuffle=is_train,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
        )
        if NUM_WORKERS > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4

        dataloader = DataLoader(feature_dataset, **kwargs)
    else:
        dataloader = dataset.train_loader if is_train else dataset.test_loader
    return dataloader


def get_dataloader_shuffle(dataset):
    dataloader = dataset.test_loader_shuffle
    return dataloader