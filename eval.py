import os
import json
import torch
import utils
from datasets.common import get_dataloader, maybe_dictionarize
from heads import get_classification_head
from modeling import ImageClassifier
from datasets.registry import get_dataset


def _eval_loop(model, dataloader, device):
    correct = torch.zeros((), device=device, dtype=torch.long)
    n = torch.zeros((), device=device, dtype=torch.long)

    with torch.inference_mode():
        for data in dataloader:
            data = maybe_dictionarize(data)
            x = data["images"].to(device, non_blocking=True)
            y = data["labels"].to(device, non_blocking=True)

            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum()
            n += y.numel()

    if int(n.item()) == 0:
        return 0.0
    return (correct.float() / n.float()).item()


def eval_single_dataset(image_encoder, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)

    top1 = _eval_loop(model, dataloader, device)

    metrics = {"top1": top1}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%")
    return metrics


def eval_single_dataset_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)

    top1 = _eval_loop(model, dataloader, device)

    metrics = {"top1": top1}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%")
    return metrics


def eval_single_dataset_preprocess_head(
    image_encoder, head, dataset_name, args, is_train=False
):
    model = ImageClassifier(image_encoder, head)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(
        dataset, is_train=is_train, args=args, image_encoder=None
    )

    top1 = _eval_loop(model, dataloader, device)

    metrics = {"top1": top1}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%")
    return metrics


def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print("Evaluating on", dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        if "top1" in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if "worst" in key or "f1" in key.lower() or "pm0" in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ":" + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, "a+") as f:
            f.write(json.dumps(info) + "\n")
        print(f"Results saved to {args.results_db}.")
    else:
        print("Results not saved (to do so, use --results_db to specify a path).")

    return info


def eval_single_dataset_preprocess_mapping_head(
    image_encoder, head, dataset_name, args, down_proj, up_proj
):
    model = ImageClassifierWithMapping(image_encoder, head, down_proj, up_proj)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)

    top1 = _eval_loop(model, dataloader, device)

    metrics = {"top1": top1}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%")
    return metrics


class ImageClassifierWithMapping(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, down_proj, up_proj):
        super().__init__()
        self.image_encoder = image_encoder

        self.down_proj = down_proj
        self.up_proj = up_proj
        self.non_linear_func = torch.nn.ReLU()
        self.classification_head = classification_head
        if self.image_encoder is not None:
            if hasattr(self.image_encoder, "train_preprocess"):
                self.train_preprocess = self.image_encoder.train_preprocess
                self.val_preprocess = self.image_encoder.val_preprocess
            elif hasattr(self.image_encoder.model, "train_preprocess"):
                self.train_preprocess = self.image_encoder.model.train_preprocess
                self.val_preprocess = self.image_encoder.model.val_preprocess

    def freeze_head(self):
        self.down_proj.weight.requires_grad_(False)
        self.up_proj.weight.requires_grad_(False)

        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        features0 = features

        features_sub = self.down_proj(features)
        features_sub = self.non_linear_func(features_sub)
        features_sub = self.up_proj(features_sub)

        features = features0 - features_sub

        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)
