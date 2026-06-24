import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DOMAINS = ["晴天", "黑天", "逆光", "雾天", "雨天"]
DOMAIN_ALIASES = {
    "晴天": ["晴天", "sunny"],
    "黑天": ["黑天", "night", "dark"],
    "逆光": ["逆光", "backlight"],
    "雾天": ["雾天", "foggy", "fog"],
    "雨天": ["雨天", "rainy", "rain"],
}
INFRARED_NAMES = {
    "ir",
    "infrared",
    "thermal",
    "gray",
    "grey",
    "hongwai",
    "红外",
    "红外图",
    "红外图像",
}
VISIBLE_NAMES = {
    "visible",
    "vis",
    "rgb",
    "color",
    "colour",
    "camera",
    "kejian",
    "可见光",
    "可见",
    "可见光图像",
}
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NUM_CLASSES = 14


def _normalized_part(part):
    return part.strip().lower().replace("-", "_").replace(" ", "_")


def _is_image(path):
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _find_domain_dir(data_root, domain):
    root = Path(data_root)
    candidates = DOMAIN_ALIASES.get(domain, [domain])
    for candidate in candidates:
        path = root / candidate
        if path.exists() and path.is_dir():
            return path

    for child in root.iterdir() if root.exists() else []:
        if child.is_dir() and child.name in candidates:
            return child

    raise FileNotFoundError(
        f"Cannot find domain directory for {domain!r} under {data_root!r}. "
        f"Tried aliases: {candidates}"
    )


def _infer_modality_and_label(parts):
    normalized = [_normalized_part(p) for p in parts]
    modality_idx = None
    modality = None

    for idx, part in enumerate(normalized[:-1]):
        if part in INFRARED_NAMES:
            modality_idx = idx
            modality = "infrared"
            break
        if part in VISIBLE_NAMES:
            modality_idx = idx
            modality = "visible"
            break

    if modality_idx is None:
        return None, None

    label = None
    if modality_idx + 1 < len(parts) - 1:
        label = parts[modality_idx + 1]
    if label is None or _normalized_part(label) in SPLIT_NAMES:
        for idx in range(modality_idx - 1, -1, -1):
            if normalized[idx] not in SPLIT_NAMES:
                label = parts[idx]
                break

    if label is None or _normalized_part(label) in INFRARED_NAMES | VISIBLE_NAMES | SPLIT_NAMES:
        return None, None

    return modality, label


def _has_explicit_splits(domain_dir):
    split_dirs = {p.name.lower() for p in domain_dir.iterdir() if p.is_dir()}
    return bool(split_dirs & SPLIT_NAMES)


def _path_matches_split(relative_parts, split):
    normalized = [_normalized_part(p) for p in relative_parts]
    if split == "val":
        return any(p in {"val", "valid", "validation"} for p in normalized)
    return split in normalized


def _collect_domain_files(data_root, domain, split=None):
    domain_dir = _find_domain_dir(data_root, domain)
    explicit_splits = _has_explicit_splits(domain_dir)
    files_by_label = defaultdict(lambda: {"infrared": [], "visible": []})

    for root, _, files in os.walk(domain_dir):
        root_path = Path(root)
        for filename in files:
            file_path = root_path / filename
            if not _is_image(file_path):
                continue

            relative_parts = file_path.relative_to(domain_dir).parts
            if explicit_splits and split is not None and not _path_matches_split(relative_parts, split):
                continue

            modality, label = _infer_modality_and_label(relative_parts)
            if modality is None:
                continue
            files_by_label[label][modality].append(str(file_path))

    for label_files in files_by_label.values():
        label_files["infrared"].sort()
        label_files["visible"].sort()

    return files_by_label


def _has_any_files(files_by_label):
    return any(files["infrared"] or files["visible"] for files in files_by_label.values())


def build_label_mapping(data_root, domains):
    labels = set()
    for domain in domains:
        files_by_label = _collect_domain_files(data_root, domain)
        labels.update(files_by_label.keys())

    if not labels:
        raise RuntimeError(
            f"No images were discovered under {data_root!r}. Expected modality directories "
            f"named like infrared/ir/红外 and visible/rgb/可见光."
        )

    numeric = True
    for label in labels:
        try:
            int(label)
        except ValueError:
            numeric = False
            break

    if numeric:
        mapping = {label: int(label) for label in labels}
        class_names = [str(i) for i in range(max(mapping.values()) + 1)]
    else:
        class_names = sorted(labels)
        mapping = {label: idx for idx, label in enumerate(class_names)}

    return mapping, class_names


def _deterministic_split(items, split, val_ratio, seed):
    grouped = defaultdict(list)
    for item in items:
        grouped[item["label"]].append(item)

    selected = []
    rng = random.Random(seed)
    for label, label_items in grouped.items():
        shuffled = list(label_items)
        rng.shuffle(shuffled)

        if len(shuffled) <= 1:
            val_count = 1 if split == "val" else 0
        else:
            val_count = max(1, int(round(len(shuffled) * val_ratio)))
            val_count = min(val_count, len(shuffled) - 1)

        if split == "val":
            selected.extend(shuffled[:val_count])
        elif split == "train":
            selected.extend(shuffled[val_count:])
        else:
            selected.extend(shuffled)

    selected.sort(key=lambda x: (x["label"], x["infrared"] or "", x["visible"] or ""))
    return selected


def _deterministic_file_split(files_by_label, split, val_ratio, seed):
    if split not in {"train", "val"}:
        return files_by_label

    selected = defaultdict(lambda: {"infrared": [], "visible": []})
    for label_name, modalities in files_by_label.items():
        for modality in ("infrared", "visible"):
            paths = list(modalities[modality])
            rng = random.Random(f"{seed}:{label_name}:{modality}")
            rng.shuffle(paths)

            if len(paths) <= 1:
                split_paths = paths if split == "train" else []
            else:
                val_count = max(1, int(round(len(paths) * val_ratio)))
                val_count = min(val_count, len(paths) - 1)
                split_paths = paths[:val_count] if split == "val" else paths[val_count:]

            selected[label_name][modality] = sorted(split_paths)

    return selected


def _pair_modalities(files_by_label, label_mapping, allow_missing_modalities, pairing_seed):
    samples = []
    for label_name, modalities in sorted(files_by_label.items(), key=lambda kv: str(kv[0])):
        label = label_mapping[label_name]
        infrared_files = modalities["infrared"]
        visible_files = modalities["visible"]

        if not infrared_files and not visible_files:
            continue
        if not allow_missing_modalities and (not infrared_files or not visible_files):
            continue

        count = max(len(infrared_files), len(visible_files))
        rng = random.Random(pairing_seed + label)
        visible_offset = rng.randrange(max(1, len(visible_files))) if visible_files else 0

        for idx in range(count):
            infrared_path = infrared_files[idx % len(infrared_files)] if infrared_files else None
            visible_path = visible_files[(idx + visible_offset) % len(visible_files)] if visible_files else None
            samples.append(
                {
                    "infrared": infrared_path,
                    "visible": visible_path,
                    "label": label,
                    "label_name": label_name,
                }
            )
    return samples


class WeatherMultiModalDataset(Dataset):
    def __init__(
        self,
        data_root,
        domain,
        split="train",
        label_mapping=None,
        transform=None,
        val_ratio=0.2,
        seed=42,
        allow_missing_modalities=False,
    ):
        self.data_root = data_root
        self.domain = domain
        self.split = split
        self.transform = transform
        self.val_ratio = val_ratio
        self.seed = seed
        self.allow_missing_modalities = allow_missing_modalities

        if label_mapping is None:
            label_mapping, _ = build_label_mapping(data_root, [domain])
        self.label_mapping = label_mapping

        explicit_files = _collect_domain_files(data_root, domain, split=split)
        all_files = _collect_domain_files(data_root, domain, split=None)
        explicit_splits = _has_explicit_splits(_find_domain_dir(data_root, domain))

        files_by_label = explicit_files if explicit_splits else all_files
        if explicit_splits and split == "val" and not _has_any_files(files_by_label):
            files_by_label = _collect_domain_files(data_root, domain, split="test")
        if not explicit_splits and split in {"train", "val"}:
            files_by_label = _deterministic_file_split(files_by_label, split, val_ratio, seed)

        samples = _pair_modalities(files_by_label, label_mapping, allow_missing_modalities, seed)

        self.samples = samples
        self.class_counts = Counter(sample["label"] for sample in samples)

        if not self.samples:
            raise RuntimeError(
                f"No usable {split} samples found for domain {domain!r}. "
                "Check modality directory names and whether each class has both infrared and visible images."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        infrared = self._load_infrared(sample["infrared"])
        visible = self._load_visible(sample["visible"])

        if self.transform:
            if isinstance(self.transform, dict):
                infrared_transform = self.transform.get("infrared")
                visible_transform = self.transform.get("visible")
                if infrared_transform:
                    infrared = infrared_transform(infrared)
                else:
                    infrared = self._to_tensor(infrared, grayscale=True)
                if visible_transform:
                    visible = visible_transform(visible)
                else:
                    visible = self._to_tensor(visible, grayscale=False)
            else:
                infrared = self.transform(infrared)
                visible = self.transform(visible)
        else:
            infrared = self._to_tensor(infrared, grayscale=True)
            visible = self._to_tensor(visible, grayscale=False)

        return {
            "infrared": infrared,
            "visible": visible,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "domain": self.domain,
            "infrared_path": sample["infrared"] or "",
            "visible_path": sample["visible"] or "",
        }

    @staticmethod
    def _to_tensor(image, grayscale):
        if grayscale:
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        else:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor

    @staticmethod
    def _load_infrared(path):
        if path is None:
            return np.zeros((224, 224), dtype=np.uint8)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read infrared image: {path}")
        return image

    @staticmethod
    def _load_visible(path):
        if path is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read visible image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _modality_paths(dataset, modality):
    return [sample[modality] for sample in dataset.samples if sample[modality]]


def _path_reference_summary(dataset, modality):
    paths = _modality_paths(dataset, modality)
    unique = set(paths)
    return {
        "references": len(paths),
        "unique": len(unique),
        "duplicate_references": len(paths) - len(unique),
    }


def _path_overlap_summary(left_dataset, right_dataset, max_examples=5):
    summary = {}
    all_left = set()
    all_right = set()
    for modality in ("infrared", "visible"):
        left_paths = set(_modality_paths(left_dataset, modality))
        right_paths = set(_modality_paths(right_dataset, modality))
        overlap = sorted(left_paths & right_paths)
        all_left.update(left_paths)
        all_right.update(right_paths)
        summary[modality] = {
            "count": len(overlap),
            "examples": overlap[:max_examples],
        }

    overlap = sorted(all_left & all_right)
    summary["any_modality"] = {
        "count": len(overlap),
        "examples": overlap[:max_examples],
    }
    return summary


def describe_dataset(dataset, class_names=None):
    counts = dict(sorted(dataset.class_counts.items()))
    readable = {}
    for class_id, count in counts.items():
        name = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
        readable[name] = count
    return {
        "domain": dataset.domain,
        "split": dataset.split,
        "num_samples": len(dataset),
        "class_counts": readable,
        "infrared_paths": _path_reference_summary(dataset, "infrared"),
        "visible_paths": _path_reference_summary(dataset, "visible"),
    }


def get_domain_data(
    data_root,
    source_domain,
    target_domain,
    batch_size,
    transform=None,
    val_ratio=0.2,
    seed=42,
    num_workers=4,
    allow_missing_modalities=False,
):
    label_mapping, class_names = build_label_mapping(data_root, [source_domain, target_domain])

    source_train = WeatherMultiModalDataset(
        data_root,
        source_domain,
        split="train",
        label_mapping=label_mapping,
        transform=transform,
        val_ratio=val_ratio,
        seed=seed,
        allow_missing_modalities=allow_missing_modalities,
    )
    source_val = WeatherMultiModalDataset(
        data_root,
        source_domain,
        split="val",
        label_mapping=label_mapping,
        transform=transform,
        val_ratio=val_ratio,
        seed=seed,
        allow_missing_modalities=allow_missing_modalities,
    )
    target_train = WeatherMultiModalDataset(
        data_root,
        target_domain,
        split="train",
        label_mapping=label_mapping,
        transform=transform,
        val_ratio=val_ratio,
        seed=seed,
        allow_missing_modalities=allow_missing_modalities,
    )
    target_val = WeatherMultiModalDataset(
        data_root,
        target_domain,
        split="val",
        label_mapping=label_mapping,
        transform=transform,
        val_ratio=val_ratio,
        seed=seed,
        allow_missing_modalities=allow_missing_modalities,
    )

    leakage_checks = {
        "source_train_vs_source_val": _path_overlap_summary(source_train, source_val),
        "target_train_vs_target_val": _path_overlap_summary(target_train, target_val),
        "source_train_vs_target_val": _path_overlap_summary(source_train, target_val),
        "source_val_vs_target_val": _path_overlap_summary(source_val, target_val),
    }

    return {
        "source_train": _make_loader(source_train, batch_size, shuffle=True, num_workers=num_workers),
        "source_val": _make_loader(source_val, batch_size, shuffle=False, num_workers=num_workers),
        "target_train": _make_loader(target_train, batch_size, shuffle=True, num_workers=num_workers),
        "target_val": _make_loader(target_val, batch_size, shuffle=False, num_workers=num_workers),
        "num_classes": max(label_mapping.values()) + 1,
        "class_names": class_names,
        "label_mapping": label_mapping,
        "dataset_stats": {
            "source_train": describe_dataset(source_train, class_names),
            "source_val": describe_dataset(source_val, class_names),
            "target_train": describe_dataset(target_train, class_names),
            "target_val": describe_dataset(target_val, class_names),
        },
        "leakage_checks": leakage_checks,
    }


WeatherDataset = WeatherMultiModalDataset


def create_data_loader(
    data_root,
    domain,
    split="train",
    batch_size=32,
    transform=None,
    label_mapping=None,
    val_ratio=0.2,
    seed=42,
    shuffle=None,
    num_workers=4,
    allow_missing_modalities=False,
):
    dataset = WeatherMultiModalDataset(
        data_root=data_root,
        domain=domain,
        split=split,
        label_mapping=label_mapping,
        transform=transform,
        val_ratio=val_ratio,
        seed=seed,
        allow_missing_modalities=allow_missing_modalities,
    )
    if shuffle is None:
        shuffle = split == "train"
    return _make_loader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
