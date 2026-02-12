import torch
import copy
from collections import defaultdict
from torch import Tensor

from NS_Merging.datasets.common import get_dataloader
from NS_Merging.datasets.registry import get_dataset


def _is_projection_eligible_param(param_name: str, param_tensor: Tensor) -> bool:
    if (
        param_name == "model.visual.class_embedding"
        or param_name == "model.visual.conv1.weight"
        or param_name == "model.visual.positional_embedding"
        or param_name == "model.visual.proj"
        or param_tensor.dim() == 1
        or "bias" in param_name
    ):
        return False
    return True


@torch.no_grad()
def _compute_taskvector_cosine_matrix(task_vectors, model) -> Tensor:
    """
    Compute cosine similarity matrix between task vectors, where each task vector is the
    concatenation of per-parameter deltas over eligible (projected) parameters.

    Returns:
        cos: Tensor[K, K] in float64
    """
    num_tasks = len(task_vectors)
    gram = torch.zeros(
        (num_tasks, num_tasks), dtype=torch.float64
    )  # sum of dot-products

    for param_name, base_param in model.named_parameters():
        if not _is_projection_eligible_param(param_name, base_param):
            continue

        rows = []
        numel = base_param.numel()
        for tv in task_vectors:
            delta = tv.vector.get(param_name, None)
            if delta is None:
                row = torch.zeros(numel, dtype=torch.float32)
            else:
                row = delta.detach().reshape(-1).to(dtype=torch.float32, device="cpu")
            rows.append(row)

        V = torch.stack(rows, dim=0)
        gram += (V @ V.t()).to(dtype=torch.float64)

    norms = torch.sqrt(torch.clamp(torch.diag(gram), min=0.0))
    denom = norms.unsqueeze(0) * norms.unsqueeze(1)
    cos = gram / torch.clamp(denom, min=1e-12)
    cos = torch.where(denom > 0, cos, torch.zeros_like(cos))
    return cos


def _ties_mask_topk_magnitudes(values: Tensor, keep_ratio: float = 0.9) -> Tensor:
    was_vector = values.dim() == 1
    if was_vector:
        values = values.unsqueeze(0)

    _, num_dims = values.shape
    if num_dims == 0:
        return values.squeeze(0) if was_vector else values

    num_keep = int(num_dims * keep_ratio)
    num_keep = max(1, min(num_keep, num_dims))
    if num_keep == num_dims:
        return values.squeeze(0) if was_vector else values

    abs_values = values.abs()
    topk_result = torch.topk(abs_values, k=num_keep, dim=1, largest=True, sorted=False)
    topk_indices: Tensor = topk_result.indices

    keep_mask = torch.zeros_like(values, dtype=torch.bool)
    keep_mask.scatter_(1, topk_indices, True)

    masked_values = values * keep_mask
    return masked_values.squeeze(0) if was_vector else masked_values


def _ties_resolve_consensus_sign(
    masked_updates: Tensor, zero_sign_policy: str = "majority"
):
    per_task_sign = torch.sign(masked_updates.sum(dim=0))

    global_majority_sign = torch.sign(per_task_sign.sum())
    if global_majority_sign == 0:
        global_majority_sign = torch.tensor(
            1, device=per_task_sign.device, dtype=per_task_sign.dtype
        )

    if zero_sign_policy == "majority":
        per_task_sign[per_task_sign == 0] = global_majority_sign
    elif zero_sign_policy == "minority":
        per_task_sign[per_task_sign == 0] = -global_majority_sign
    return per_task_sign


def _ties_select_disjoint(masked_updates: Tensor, consensus_sign: Tensor):
    sign_compatible_mask = torch.where(
        consensus_sign.unsqueeze(0) > 0, masked_updates > 0, masked_updates < 0
    )
    return masked_updates * sign_compatible_mask


def _apply_ties_to_parameter_across_tasks(
    task_vectors,
    param_name,
    reset_ratio,
    preserve_norm=True,
    zero_sign_policy="majority",
):
    param_vectors = []
    reference_tensor = None
    for task_vector in task_vectors:
        delta = task_vector.vector.get(param_name, None)
        param_vectors.append(delta)
        if delta is not None and reference_tensor is None:
            reference_tensor = delta

    if reference_tensor is None:
        return

    ref_dtype = reference_tensor.dtype
    ref_device = reference_tensor.device

    flat_deltas = []
    original_norms = []
    for delta in param_vectors:
        if delta is None:
            flat_delta = torch.zeros_like(reference_tensor).view(-1)
            original_norms.append(torch.tensor(0.0, device=ref_device))
        else:
            flat_delta = delta.view(-1)
            original_norms.append(torch.norm(flat_delta.float(), p=2))
        flat_deltas.append(flat_delta)

    stacked_deltas = torch.stack(flat_deltas, dim=0).float()

    masked_deltas = _ties_mask_topk_magnitudes(stacked_deltas, keep_ratio=reset_ratio)
    consensus_sign = _ties_resolve_consensus_sign(
        masked_deltas, zero_sign_policy=zero_sign_policy
    )
    disjoint_selected = _ties_select_disjoint(masked_deltas, consensus_sign)

    for task_idx, task_vector in enumerate(task_vectors):
        new_flat_delta = disjoint_selected[task_idx].to(ref_dtype)

        if preserve_norm:
            new_norm = torch.norm(new_flat_delta.float(), p=2)
            if new_norm > 0 and original_norms[task_idx] > 0:
                new_flat_delta = new_flat_delta * (
                    original_norms[task_idx].to(new_flat_delta.device) / new_norm
                )

        task_vector.vector[param_name] = new_flat_delta.view_as(reference_tensor)


def _should_skip_param_for_projection(param_name, task_vector):
    if task_vector.vector[param_name] is None:
        return True
    if (
        param_name == "model.visual.class_embedding"
        or param_name == "model.visual.conv1.weight"
        or param_name == "model.visual.positional_embedding"
        or param_name == "model.visual.proj"
        or len(task_vector.vector[param_name].shape) == 1
        or "bias" in param_name
    ):
        return True
    return False


def _project_weight_to_right_nullspace(weight, activations, rcond=0):
    if activations is None:
        return weight
    if activations.numel() == 0:
        return weight

    weight_dtype = weight.dtype
    device = weight.device

    weight_fp32 = weight.to(device=device, dtype=torch.float32)
    activations_fp32 = activations.to(device=device, dtype=torch.float32)

    num_samples = activations_fp32.shape[0]
    if num_samples == 0:
        return weight

    gram = activations_fp32.mm(activations_fp32.t())
    gram_pinv = torch.linalg.pinv(gram, rcond=rcond)

    weight_x_t = weight_fp32.mm(activations_fp32.t())  # (m x n)
    correction = weight_x_t.mm(gram_pinv).mm(activations_fp32)  # (m x d)

    projected_weight = weight_fp32 - correction

    orig_norm = torch.norm(weight_fp32)
    proj_norm = torch.norm(projected_weight)
    if proj_norm > 0:
        projected_weight = projected_weight * (orig_norm / proj_norm)

    return projected_weight.to(dtype=weight_dtype)


def ns_merging(
    args, task_vectors, pretrained_checkpoint, exam_datasets, cur_task_vectors=None
):
    if cur_task_vectors is None:
        cur_task_vectors = copy.deepcopy(task_vectors)
    result_task_vectors = copy.deepcopy(task_vectors)

    num_ns_cycles = getattr(args, "ns_cycles", 8)  # 8个任务

    per_task_features = []
    model = None

    for task_idx, dataset_name in enumerate(exam_datasets):
        task_vector = cur_task_vectors[task_idx]
        model = task_vector.apply_to(
            pretrained_checkpoint, scaling_coef=args.scaling_coef_
        ).to("cuda")

        dataset = get_dataset(
            dataset_name,
            model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        dataloader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None
        )

        module_input_cache = defaultdict(list)

        def forward_hook(module, input, output):
            x = input[0] if isinstance(input, (tuple, list)) else input
            module_input_cache[module].append(x.detach().cpu())

        forward_hooks = []
        named_module_map = dict(model.named_modules())
        hooked_modules = set()
        for param_name, param in model.named_parameters():
            module_name = ".".join(param_name.split(".")[:-1])
            if not module_name:
                continue
            if module_name in hooked_modules:
                continue

            module = named_module_map[module_name]
            forward_hooks.append(module.register_forward_hook(forward_hook))
            hooked_modules.add(module_name)

        num_collected_samples = 0
        for batch_idx, batch in enumerate(dataloader):
            images = batch[0].to("cuda")
            if images.shape[0] > args.exp_size:
                images = images[: args.exp_size, ...]

            with torch.no_grad():
                model(images)

            num_collected_samples += images.shape[0]
            if num_collected_samples >= args.exp_size:
                break

        for hook in forward_hooks:
            hook.remove()

        features_by_param = {}
        model.to("cpu")

        for param_name, param in model.named_parameters():
            module_name = ".".join(param_name.split(".")[:-1])
            if not module_name:
                continue
            module = dict(model.named_modules())[module_name]
            cached_inputs = module_input_cache.get(module)

            if cached_inputs is None:
                features_by_param[param_name] = None
                continue

            if (
                param_name == "model.visual.class_embedding"
                or param_name == "model.visual.conv1.weight"
                or param_name == "model.visual.positional_embedding"
                or param_name == "model.visual.proj"
                or len(task_vector.vector[param_name].shape) == 1
                or "bias" in param_name
            ):
                continue

            input_activations = torch.cat(cached_inputs, dim=0)
            input_activations = input_activations.view(-1, input_activations.shape[-1])
            features_by_param[param_name] = input_activations

        per_task_features.append(features_by_param)

    if getattr(args, "ratio", 1) != 0:
        # ===== cosine BEFORE projection =====
        if getattr(args, "log_ns_cosine", False):
            cos_before = _compute_taskvector_cosine_matrix(result_task_vectors, model)
            print("\n[NS][COSINE] BEFORE projection (task x task):\n", cos_before)

        num_tasks = len(result_task_vectors)

        for _ in range(num_ns_cycles):
            for target_task_idx in range(num_tasks):
                target_task_vector = result_task_vectors[target_task_idx]
                for param_name, param in list(model.named_parameters()):
                    if _should_skip_param_for_projection(
                        param_name, target_task_vector
                    ):
                        continue

                    target_activations = per_task_features[target_task_idx].get(
                        param_name, None
                    )
                    if target_activations is None:
                        continue

                    for source_task_idx in range(num_tasks):
                        if source_task_idx == target_task_idx:
                            continue
                        source_task_vector = result_task_vectors[source_task_idx]
                        if _should_skip_param_for_projection(
                            param_name, source_task_vector
                        ):
                            continue

                        weight_delta = source_task_vector.vector[param_name]
                        if weight_delta is None:
                            continue

                        source_task_vector.vector[param_name] = (
                            _project_weight_to_right_nullspace(
                                weight=weight_delta, activations=target_activations
                            )
                        )
        # ===== cosine AFTER projection =====
        if getattr(args, "log_ns_cosine", False):
            cos_after = _compute_taskvector_cosine_matrix(result_task_vectors, model)
            print("\n[NS][COSINE] AFTER projection (task x task):\n", cos_after)

    ln_weight_reset_ratio = getattr(args, "ties_reset_thresh_ln", None)
    if ln_weight_reset_ratio is not None and ln_weight_reset_ratio != 0:
        preserve_ln_norm = getattr(args, "ties_keep_norm_ln", True)
        zero_sign_policy = getattr(args, "ties_zero_sign_method", "majority")

        for param_name, param in list(model.named_parameters()):
            if "ln" in param_name and "weight" in param_name:
                if "ln_post" in param_name:
                    continue
                if result_task_vectors[0].vector.get(param_name, None) is None:
                    continue

                _apply_ties_to_parameter_across_tasks(
                    result_task_vectors,
                    param_name=param_name,
                    reset_ratio=ln_weight_reset_ratio,
                    preserve_norm=preserve_ln_norm,
                    zero_sign_policy=zero_sign_policy,
                )

    bias_reset_ratio = getattr(args, "ties_reset_thresh_bias", None)
    if bias_reset_ratio is not None and bias_reset_ratio != 0:
        preserve_bias_norm = getattr(args, "ties_keep_norm_bias", False)
        zero_sign_policy = getattr(args, "ties_zero_sign_method", "majority")

        for param_name, param in list(model.named_parameters()):
            if "ln" in param_name and "weight" in param_name:
                continue

            reference_delta = result_task_vectors[0].vector.get(param_name, None)
            if reference_delta is None:
                continue

            if (
                param_name == "model.visual.class_embedding"
                or param_name == "model.visual.conv1.weight"
                or param_name == "model.visual.positional_embedding"
                or param_name == "model.visual.proj"
                or len(reference_delta.shape) == 1
                or "bias" in param_name
            ):
                _apply_ties_to_parameter_across_tasks(
                    result_task_vectors,
                    param_name=param_name,
                    reset_ratio=bias_reset_ratio,
                    preserve_norm=preserve_bias_norm,
                    zero_sign_policy=zero_sign_policy,
                )

    return result_task_vectors
