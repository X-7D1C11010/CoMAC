import argparse
import copy
import json
import logging
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.weather_dataset import DOMAINS, get_domain_data
from models.weather_comac import CoMACWeatherClassifier, update_ema_model
from utils.metrics import SegmentationMetrics
from utils.utils import save_checkpoint, set_seed


DOMAIN_MAP = {
    "晴天": "sunny",
    "黑天": "night",
    "逆光": "backlight",
    "雾天": "foggy",
    "雨天": "rainy",
}


def make_transforms(image_size):
    return {
        "infrared": transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
        "visible": transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    }


def make_criterion(label_smoothing, logger):
    try:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    except TypeError:
        if label_smoothing > 0:
            logger.warning("Current PyTorch does not support label_smoothing; falling back to plain CrossEntropyLoss.")
        return nn.CrossEntropyLoss()


def resolve_device(args, logger):
    requested = args.device.lower()
    if requested == "cpu":
        logger.info("Using device: cpu")
        return torch.device("cpu")

    if args.gpu_id is not None:
        requested = f"cuda:{args.gpu_id}"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

        if requested == "cuda":
            device_index = torch.cuda.current_device()
        else:
            try:
                device_index = int(requested.split(":", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid CUDA device string: {args.device!r}") from exc

        device_count = torch.cuda.device_count()
        if device_index < 0 or device_index >= device_count:
            raise ValueError(f"Requested GPU {device_index}, but only {device_count} CUDA device(s) are visible.")

        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(device_index)})")
        return device

    raise ValueError("Unsupported --device value. Use 'cpu', 'cuda', or 'cuda:<id>'.")


def to_device(batch, device):
    return {
        "infrared": batch["infrared"].to(device, non_blocking=True),
        "visible": batch["visible"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
    }


def masked_cross_entropy(logits, labels, mask):
    if mask is None:
        return F.cross_entropy(logits, labels)
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], labels[mask])


def class_histogram(preds, labels, num_classes):
    pred_hist = torch.bincount(preds.detach().cpu(), minlength=num_classes).tolist()
    label_hist = torch.bincount(labels.detach().cpu(), minlength=num_classes).tolist()
    return pred_hist, label_hist


def log_leakage_checks(logger, leakage_checks):
    logger.info("Path leakage checks:\n" + json.dumps(leakage_checks, ensure_ascii=False, indent=2))
    for check_name, summary in leakage_checks.items():
        overlaps = []
        for modality, item in summary.items():
            count = item.get("count", 0)
            if count > 0:
                overlaps.append(f"{modality}={count}")
        if overlaps:
            logger.warning(f"Possible data leakage in {check_name}: " + ", ".join(overlaps))


@torch.no_grad()
def compute_centroids(model, data_loader, num_classes, feature_dim, device):
    model.eval()
    sums_ir = torch.zeros(num_classes, feature_dim, device=device)
    sums_vis = torch.zeros(num_classes, feature_dim, device=device)
    counts = torch.zeros(num_classes, device=device)

    for batch in data_loader:
        batch = to_device(batch, device)
        output = model(batch["infrared"], batch["visible"], return_features=True)
        labels = batch["label"]
        for class_id in labels.unique():
            mask = labels == class_id
            idx = int(class_id.item())
            sums_ir[idx] += torch.nn.functional.normalize(output["features_ir"][mask], dim=1).sum(dim=0)
            sums_vis[idx] += torch.nn.functional.normalize(output["features_vis"][mask], dim=1).sum(dim=0)
            counts[idx] += mask.sum()

    centroids_ir = torch.zeros_like(sums_ir)
    centroids_vis = torch.zeros_like(sums_vis)
    valid = counts > 0
    centroids_ir[valid] = torch.nn.functional.normalize(sums_ir[valid] / counts[valid].unsqueeze(1), dim=1)
    centroids_vis[valid] = torch.nn.functional.normalize(sums_vis[valid] / counts[valid].unsqueeze(1), dim=1)
    return centroids_ir, centroids_vis, counts.detach().cpu().numpy().astype(int).tolist()


@torch.no_grad()
def evaluate(model, data_loader, device, num_classes):
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes)
    prediction_hist = Counter()
    label_hist = Counter()
    modal_weight_sum = torch.zeros(2)
    total_samples = 0

    for batch in data_loader:
        batch = to_device(batch, device)
        output = model(batch["infrared"], batch["visible"], return_features=True)
        preds = torch.argmax(output["logits"], dim=1)
        metrics.update(preds, batch["label"])

        pred_hist, gt_hist = class_histogram(preds, batch["label"], num_classes)
        prediction_hist.update({idx: count for idx, count in enumerate(pred_hist) if count})
        label_hist.update({idx: count for idx, count in enumerate(gt_hist) if count})

        modal_weight_sum += output["modal_weights"].detach().cpu().sum(dim=0)
        total_samples += preds.numel()

    results = metrics.compute()
    results["pred_hist"] = dict(sorted(prediction_hist.items()))
    results["label_hist"] = dict(sorted(label_hist.items()))
    if total_samples > 0:
        avg_weights = (modal_weight_sum / total_samples).tolist()
        results["avg_modal_weight_ir"] = float(avg_weights[0])
        results["avg_modal_weight_visible"] = float(avg_weights[1])
    else:
        results["avg_modal_weight_ir"] = 0.5
        results["avg_modal_weight_visible"] = 0.5
    return results


def train_source(model, data_loader, criterion, optimizer, device, epoch, args, logger, num_classes):
    model.train()
    total_loss = 0.0
    total_samples = 0
    metrics = SegmentationMetrics(num_classes=num_classes)

    for batch_idx, raw_batch in enumerate(data_loader):
        batch = to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)

        output = model(batch["infrared"], batch["visible"], return_features=True)
        loss_fused = criterion(output["logits"], batch["label"])
        loss_ir = criterion(output["logits_ir"], batch["label"])
        loss_vis = criterion(output["logits_vis"], batch["label"])
        loss = loss_fused + args.modal_loss_weight * (loss_ir + loss_vis)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            model.update_centroids(
                output["features_ir"].detach(),
                output["features_vis"].detach(),
                batch["label"],
                momentum=args.centroid_momentum,
            )

        preds = torch.argmax(output["logits"], dim=1)
        metrics.update(preds, batch["label"])
        total_loss += loss.item() * batch["label"].size(0)
        total_samples += batch["label"].size(0)

        if (batch_idx + 1) % args.log_interval == 0:
            logger.info(
                f"Source Epoch [{epoch}], Batch [{batch_idx + 1}/{len(data_loader)}], "
                f"Loss: {loss.item():.4f}, Acc: {metrics.compute_accuracy():.4f}"
            )

    results = metrics.compute()
    avg_loss = total_loss / max(1, total_samples)
    logger.info(
        f"Source Training - Epoch [{epoch}], Loss: {avg_loss:.4f}, "
        f"Acc: {results['accuracy']:.4f}, F1: {results['f1_score']:.4f}"
    )
    return avg_loss, results


def augment_batch(infrared, visible, aug_idx):
    infrared_aug = infrared
    visible_aug = visible

    if aug_idx % 2 == 0:
        infrared_aug = torch.flip(infrared_aug, dims=[-1])
        visible_aug = torch.flip(visible_aug, dims=[-1])
    if aug_idx % 3 == 1:
        infrared_aug = infrared_aug + torch.randn_like(infrared_aug) * 0.03
        visible_aug = visible_aug + torch.randn_like(visible_aug) * 0.03
    if aug_idx % 3 == 2:
        infrared_aug = infrared_aug * 0.95
        visible_aug = visible_aug * 1.05

    return infrared_aug, visible_aug


@torch.no_grad()
def averaged_teacher_output(teacher, infrared, visible, num_augmentations):
    if num_augmentations <= 0:
        return teacher(infrared, visible, return_features=True)

    accumulator = None
    for aug_idx in range(num_augmentations):
        infrared_aug, visible_aug = augment_batch(infrared, visible, aug_idx)
        output = teacher(infrared_aug, visible_aug, return_features=True)
        if accumulator is None:
            accumulator = {key: value.detach().clone() for key, value in output.items() if torch.is_tensor(value)}
        else:
            for key in accumulator:
                accumulator[key] += output[key].detach()

    for key in accumulator:
        accumulator[key] /= float(num_augmentations)
    return accumulator


@torch.no_grad()
def intra_modal_aggregate(teacher, raw_logits, raw_features, aug_logits, aug_features, modality):
    centroids = teacher.centroids_ir if modality == "infrared" else teacher.centroids_vis
    raw_weight = teacher.modality_reliability(raw_logits, raw_features, centroids)
    aug_weight = teacher.modality_reliability(aug_logits, aug_features, centroids)
    denom = (raw_weight + aug_weight).clamp_min(1e-6).unsqueeze(1)
    logits = (raw_weight.unsqueeze(1) * raw_logits + aug_weight.unsqueeze(1) * aug_logits) / denom
    features = (raw_weight.unsqueeze(1) * raw_features + aug_weight.unsqueeze(1) * aug_features) / denom
    return logits, features


@torch.no_grad()
def generate_pseudo_labels(teacher, infrared, visible, args):
    teacher.eval()
    raw = teacher(infrared, visible, return_features=True)
    augmented = averaged_teacher_output(teacher, infrared, visible, args.num_augmentations)

    logits_ir, features_ir = intra_modal_aggregate(
        teacher,
        raw["logits_ir"],
        raw["features_ir"],
        augmented["logits_ir"],
        augmented["features_ir"],
        "infrared",
    )
    logits_vis, features_vis = intra_modal_aggregate(
        teacher,
        raw["logits_vis"],
        raw["features_vis"],
        augmented["logits_vis"],
        augmented["features_vis"],
        "visible",
    )

    reliability_ir = teacher.modality_reliability(logits_ir, features_ir, teacher.centroids_ir)
    reliability_vis = teacher.modality_reliability(logits_vis, features_vis, teacher.centroids_vis)
    modal_weights = torch.softmax(
        torch.stack([reliability_ir, reliability_vis], dim=1) / args.reliability_temperature,
        dim=1,
    )

    probs_ir = torch.softmax(logits_ir, dim=1)
    probs_vis = torch.softmax(logits_vis, dim=1)
    fused_probs = modal_weights[:, 0:1] * probs_ir + modal_weights[:, 1:2] * probs_vis
    confidence, pseudo_labels = fused_probs.max(dim=1)
    mask = confidence >= args.confidence_threshold

    return {
        "pseudo_labels": pseudo_labels,
        "confidence": confidence,
        "mask": mask,
        "modal_weights": modal_weights,
    }


def next_source_batch(source_iter, source_loader):
    try:
        return source_iter, next(source_iter)
    except StopIteration:
        source_iter = iter(source_loader)
        return source_iter, next(source_iter)


def adapt_target_comac(
    model,
    teacher,
    source_loader,
    target_train_loader,
    target_val_loader,
    optimizer,
    criterion,
    device,
    args,
    logger,
    num_classes,
):
    best_results = None
    best_epoch = 0
    source_iter = iter(source_loader)

    for epoch in range(1, args.adapt_epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        pseudo_kept = 0
        pseudo_total = 0
        pseudo_confidence = 0.0

        for raw_target in target_train_loader:
            target = to_device(raw_target, device)
            optimizer.zero_grad(set_to_none=True)

            pseudo = generate_pseudo_labels(teacher, target["infrared"], target["visible"], args)
            output = model(target["infrared"], target["visible"], return_features=True)
            mask = pseudo["mask"]
            pseudo_labels = pseudo["pseudo_labels"]

            target_loss = (
                masked_cross_entropy(output["logits"], pseudo_labels, mask)
                + args.modal_loss_weight * masked_cross_entropy(output["logits_ir"], pseudo_labels, mask)
                + args.modal_loss_weight * masked_cross_entropy(output["logits_vis"], pseudo_labels, mask)
            )
            if mask.any():
                center_loss = 0.5 * (
                    model.center_contrastive_loss(output["features_ir"][mask], pseudo_labels[mask], "infrared", args.center_temperature)
                    + model.center_contrastive_loss(output["features_vis"][mask], pseudo_labels[mask], "visible", args.center_temperature)
                )
            else:
                center_loss = output["logits"].sum() * 0.0

            loss = args.target_loss_weight * target_loss + args.center_loss_weight * center_loss

            if args.target_supervised_weight > 0:
                target_supervised_loss = (
                    criterion(output["logits"], target["label"])
                    + args.modal_loss_weight * criterion(output["logits_ir"], target["label"])
                    + args.modal_loss_weight * criterion(output["logits_vis"], target["label"])
                )
                loss = loss + args.target_supervised_weight * target_supervised_loss

            if args.source_replay_weight > 0:
                source_iter, raw_source = next_source_batch(source_iter, source_loader)
                source = to_device(raw_source, device)
                source_output = model(source["infrared"], source["visible"], return_features=True)
                source_loss = (
                    criterion(source_output["logits"], source["label"])
                    + args.modal_loss_weight * criterion(source_output["logits_ir"], source["label"])
                    + args.modal_loss_weight * criterion(source_output["logits_vis"], source["label"])
                )
                loss = loss + args.source_replay_weight * source_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                if mask.any():
                    model.update_centroids(
                        output["features_ir"].detach(),
                        output["features_vis"].detach(),
                        pseudo_labels,
                        mask=mask,
                        momentum=args.centroid_momentum,
                    )
                if random.random() < args.centroid_restore_prob:
                    model.restore_source_centroids(args.centroid_restore_strength)
                model.stochastic_restore(args.weight_restore_rate)
                update_ema_model(model, teacher, momentum=args.teacher_momentum)

            batch_size = target["label"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            pseudo_kept += int(mask.sum().item())
            pseudo_total += int(mask.numel())
            pseudo_confidence += float(pseudo["confidence"].sum().item())

        avg_loss = total_loss / max(1, total_samples)
        coverage = pseudo_kept / max(1, pseudo_total)
        avg_confidence = pseudo_confidence / max(1, pseudo_total)
        logger.info(
            f"CoMAC Adapt - Epoch [{epoch}], Loss: {avg_loss:.4f}, "
            f"Pseudo Coverage: {coverage:.4f}, Pseudo Conf: {avg_confidence:.4f}"
        )

        results = evaluate(model, target_val_loader, device, num_classes)
        logger.info(
            f"Target Val - Epoch [{epoch}], Acc: {results['accuracy']:.4f}, "
            f"Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, "
            f"F1: {results['f1_score']:.4f}, "
            f"ModalW(IR/VIS): {results['avg_modal_weight_ir']:.3f}/{results['avg_modal_weight_visible']:.3f}"
        )

        if best_results is None or results["accuracy"] > best_results["accuracy"]:
            best_results = results
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, args.current_output_dir, f"best_target_{args.current_seed}.pth")

    return best_epoch, best_results


def run_single_experiment(source_domain, target_domain, data_root, output_dir, seed, args, logger=None):
    set_seed(seed)
    device = resolve_device(args, logger)

    transform = make_transforms(args.image_size)
    data_loaders = get_domain_data(
        data_root,
        source_domain,
        target_domain,
        args.batch_size,
        transform=transform,
        val_ratio=args.val_ratio,
        seed=seed,
        num_workers=args.num_workers,
        allow_missing_modalities=args.allow_missing_modalities,
    )

    num_classes = data_loaders["num_classes"]
    logger.info("Dataset stats:\n" + json.dumps(data_loaders["dataset_stats"], ensure_ascii=False, indent=2))
    log_leakage_checks(logger, data_loaders["leakage_checks"])
    logger.info(f"Discovered num_classes: {num_classes}")

    target_val_size = len(data_loaders["target_val"].dataset)
    if target_val_size < 100:
        logger.warning(
            f"Target validation split has only {target_val_size} samples; best-epoch metrics can be unstable."
        )
    if args.target_supervised_weight > 0:
        logger.warning(
            "target_supervised_weight > 0 uses target_train labels during adaptation. "
            "Target Val and best_target metrics are supervised fine-tuning validation results; "
            "set --target_supervised_weight 0 for label-free CoMAC-style adaptation."
        )

    model = CoMACWeatherClassifier(
        num_classes=num_classes,
        backbone=args.backbone,
        feature_dim=args.feature_dim,
        pretrained=args.pretrained,
        reliability_temperature=args.reliability_temperature,
    ).to(device)

    criterion = make_criterion(args.label_smoothing, logger)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.source_epochs + args.adapt_epochs))

    best_source_results = None
    best_source_epoch = 0
    logger.info(f"=== Source Training (seed {seed}) ===")
    for epoch in range(1, args.source_epochs + 1):
        train_source(model, data_loaders["source_train"], criterion, optimizer, device, epoch, args, logger, num_classes)
        source_val = evaluate(model, data_loaders["source_val"], device, num_classes)
        scheduler.step()
        logger.info(
            f"Source Val - Epoch [{epoch}], Acc: {source_val['accuracy']:.4f}, "
            f"Precision: {source_val['precision']:.4f}, Recall: {source_val['recall']:.4f}, "
            f"F1: {source_val['f1_score']:.4f}"
        )
        if best_source_results is None or source_val["accuracy"] > best_source_results["accuracy"]:
            best_source_results = source_val
            best_source_epoch = epoch
            save_checkpoint(model, optimizer, epoch, output_dir, f"best_source_{seed}.pth")

    centroids_ir, centroids_vis, centroid_counts = compute_centroids(
        model,
        data_loaders["source_train"],
        num_classes,
        args.feature_dim,
        device,
    )
    model.set_centroids(centroids_ir, centroids_vis, set_source=True)
    model.snapshot_source_state()
    logger.info(f"Source centroid counts: {centroid_counts}")

    source_final = evaluate(model, data_loaders["source_val"], device, num_classes)
    target_before = evaluate(model, data_loaders["target_val"], device, num_classes)
    logger.info(
        f"Before Adapt - Source Acc: {source_final['accuracy']:.4f}, "
        f"Target Acc: {target_before['accuracy']:.4f}, Target F1: {target_before['f1_score']:.4f}, "
        f"Target PredHist: {target_before['pred_hist']}"
    )

    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    args.current_output_dir = output_dir
    args.current_seed = seed
    logger.info(f"=== CoMAC-inspired Target Adaptation (seed {seed}) ===")
    best_target_epoch, best_target_results = adapt_target_comac(
        model,
        teacher,
        data_loaders["source_train"],
        data_loaders["target_train"],
        data_loaders["target_val"],
        optimizer,
        criterion,
        device,
        args,
        logger,
        num_classes,
    )

    target_after = evaluate(model, data_loaders["target_val"], device, num_classes)
    logger.info(
        f"After Adapt - Target Acc: {target_after['accuracy']:.4f}, "
        f"F1: {target_after['f1_score']:.4f}, PredHist: {target_after['pred_hist']}, "
        f"LabelHist: {target_after['label_hist']}"
    )

    return {
        "seed": seed,
        "num_classes": num_classes,
        "best_source_epoch": best_source_epoch,
        "best_source_results": best_source_results,
        "target_before_results": target_before,
        "best_target_epoch": best_target_epoch,
        "best_target_results": best_target_results,
        "target_after_results": target_after,
    }


def _serializable_results(results):
    def convert(value):
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    return json.loads(json.dumps(results, default=convert))


def run_experiments(source_domain, target_domains, data_root, output_root, args):
    os.makedirs(output_root, exist_ok=True)
    all_results = {}

    for target_domain in target_domains:
        src_en = DOMAIN_MAP.get(source_domain, source_domain)
        tgt_en = DOMAIN_MAP.get(target_domain, target_domain)
        log_path = os.path.join(output_root, f"{src_en}_to_{tgt_en}.log")
        domain_output_dir = os.path.join(output_root, f"{src_en}_to_{tgt_en}")
        os.makedirs(domain_output_dir, exist_ok=True)

        logger = logging.getLogger(f"{src_en}_to_{tgt_en}")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        console_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info(f"========== Experiment: {source_domain} -> {target_domain} ==========")
        logger.info(f"Runs: {args.num_runs}")
        logger.info(f"Source epochs: {args.source_epochs}, Adapt epochs: {args.adapt_epochs}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Args: {args}")

        run_results = []
        for run_idx in range(args.num_runs):
            seed = args.seed + run_idx
            logger.info(f"\n---------- Run {run_idx + 1}/{args.num_runs} (seed {seed}) ----------")
            result = run_single_experiment(source_domain, target_domain, data_root, domain_output_dir, seed, args, logger)
            result = _serializable_results(result)
            run_results.append(result)
            logger.info(
                f"Run {run_idx + 1} done. Source best epoch {result['best_source_epoch']}, "
                f"Acc {result['best_source_results']['accuracy']:.4f}; "
                f"Target best epoch {result['best_target_epoch']}, "
                f"Acc {result['best_target_results']['accuracy']:.4f}, "
                f"F1 {result['best_target_results']['f1_score']:.4f}; "
                f"Target final Acc {result['target_after_results']['accuracy']:.4f}, "
                f"F1 {result['target_after_results']['f1_score']:.4f}"
            )

        all_results[target_domain] = run_results

        acc_list = [r["best_target_results"]["accuracy"] for r in run_results]
        prec_list = [r["best_target_results"]["precision"] for r in run_results]
        rec_list = [r["best_target_results"]["recall"] for r in run_results]
        f1_list = [r["best_target_results"]["f1_score"] for r in run_results]
        final_acc_list = [r["target_after_results"]["accuracy"] for r in run_results]
        final_prec_list = [r["target_after_results"]["precision"] for r in run_results]
        final_rec_list = [r["target_after_results"]["recall"] for r in run_results]
        final_f1_list = [r["target_after_results"]["f1_score"] for r in run_results]

        best_statistics = {
            "accuracy": {"mean": float(np.mean(acc_list)), "std": float(np.std(acc_list))},
            "precision": {"mean": float(np.mean(prec_list)), "std": float(np.std(prec_list))},
            "recall": {"mean": float(np.mean(rec_list)), "std": float(np.std(rec_list))},
            "f1_score": {"mean": float(np.mean(f1_list)), "std": float(np.std(f1_list))},
        }
        final_statistics = {
            "accuracy": {"mean": float(np.mean(final_acc_list)), "std": float(np.std(final_acc_list))},
            "precision": {"mean": float(np.mean(final_prec_list)), "std": float(np.std(final_prec_list))},
            "recall": {"mean": float(np.mean(final_rec_list)), "std": float(np.std(final_rec_list))},
            "f1_score": {"mean": float(np.mean(final_f1_list)), "std": float(np.std(final_f1_list))},
        }

        summary = {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "num_runs": args.num_runs,
            "source_epochs": args.source_epochs,
            "adapt_epochs": args.adapt_epochs,
            "batch_size": args.batch_size,
            "results": run_results,
            "statistics": best_statistics,
            "best_statistics": best_statistics,
            "final_statistics": final_statistics,
        }

        logger.info(f"\n========== {source_domain} -> {target_domain} Summary ==========")
        logger.info(f"Best ACC: mean = {best_statistics['accuracy']['mean']:.4f}, std = {best_statistics['accuracy']['std']:.4f}")
        logger.info(f"Best Precision: mean = {best_statistics['precision']['mean']:.4f}, std = {best_statistics['precision']['std']:.4f}")
        logger.info(f"Best Recall: mean = {best_statistics['recall']['mean']:.4f}, std = {best_statistics['recall']['std']:.4f}")
        logger.info(f"Best F1: mean = {best_statistics['f1_score']['mean']:.4f}, std = {best_statistics['f1_score']['std']:.4f}")
        logger.info(f"Final ACC: mean = {final_statistics['accuracy']['mean']:.4f}, std = {final_statistics['accuracy']['std']:.4f}")
        logger.info(f"Final Precision: mean = {final_statistics['precision']['mean']:.4f}, std = {final_statistics['precision']['std']:.4f}")
        logger.info(f"Final Recall: mean = {final_statistics['recall']['mean']:.4f}, std = {final_statistics['recall']['std']:.4f}")
        logger.info(f"Final F1: mean = {final_statistics['f1_score']['mean']:.4f}, std = {final_statistics['f1_score']['std']:.4f}")

        with open(os.path.join(domain_output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    overall_summary = {
        "source_domain": source_domain,
        "target_domains": target_domains,
        "num_runs": args.num_runs,
        "source_epochs": args.source_epochs,
        "adapt_epochs": args.adapt_epochs,
        "batch_size": args.batch_size,
        "all_results": all_results,
    }
    with open(os.path.join(output_root, "overall_summary.json"), "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="CoMAC-inspired infrared/visible weather-domain classification")
    parser.add_argument("--data_root", type=str, default="/home/lixiang/lx/Data")
    parser.add_argument("--output_root", type=str, default="/home/lixiang/lx/CoMAC/experiments")
    parser.add_argument("--source_domain", type=str, default="晴天")
    parser.add_argument("--target_domains", type=str, nargs="+", default=["黑天", "逆光", "雾天", "雨天"])
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=50, help="Legacy shortcut used when source/adapt epochs are omitted")
    parser.add_argument("--source_epochs", type=int, default=None)
    parser.add_argument("--adapt_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", help="Use cpu, cuda, or cuda:<id>.")
    parser.add_argument("--gpu_id", type=int, default=None, help="CUDA GPU index to use. Overrides --device when set.")

    parser.add_argument("--backbone", type=str, default="resnet18", choices=["cnn", "resnet18", "resnet34"])
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights when available")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--num_augmentations", type=int, default=3)
    parser.add_argument("--teacher_momentum", type=float, default=0.999)
    parser.add_argument("--centroid_momentum", type=float, default=0.95)
    parser.add_argument("--centroid_restore_prob", type=float, default=0.3)
    parser.add_argument("--centroid_restore_strength", type=float, default=0.1)
    parser.add_argument("--weight_restore_rate", type=float, default=0.001)
    parser.add_argument("--target_loss_weight", type=float, default=1.0)
    parser.add_argument("--target_supervised_weight", type=float, default=0.0)
    parser.add_argument("--source_replay_weight", type=float, default=0.3)
    parser.add_argument("--center_loss_weight", type=float, default=0.1)
    parser.add_argument("--center_temperature", type=float, default=0.1)
    parser.add_argument("--modal_loss_weight", type=float, default=0.3)
    parser.add_argument("--reliability_temperature", type=float, default=0.5)
    parser.add_argument("--allow_missing_modalities", action="store_true")
    parser.add_argument("--log_interval", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.source_epochs is None:
        args.source_epochs = args.num_epochs
    if args.adapt_epochs is None:
        args.adapt_epochs = args.num_epochs

    run_experiments(
        source_domain=args.source_domain,
        target_domains=args.target_domains,
        data_root=args.data_root,
        output_root=args.output_root,
        args=args,
    )
