#!/usr/bin/env python3
"""Generate a strong Office-Home class-incremental protocol.

Protocol:
1. Scan the original Office-Home directory tree:
      <root>/<domain>/<class_name>/<image_name>
2. Split each (domain, class) bucket into train/test subsets with a fixed seed.
3. Shuffle all 65 classes with a fixed seed.
4. Split shuffled classes into tasks by init_cls/increment.
5. Assign exactly one training domain to each task in a cyclic shuffled order.
6. Build:
   - train.txt: only the assigned-domain train subset for each task class
   - test.txt: all held-out test subsets for all selected classes across all domains

The output txt format matches the loader expectation:
    relative/path/to/image.jpg label
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    relpath: str
    domain: str
    class_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a strong Office-Home train/test split."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Office-Home root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write train.txt/test.txt/metadata.json.",
    )
    parser.add_argument(
        "--init-cls",
        type=int,
        required=True,
        help="Number of classes in the first task.",
    )
    parser.add_argument(
        "--increment",
        type=int,
        required=True,
        help="Number of classes in each following task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1993,
        help="Random seed for class shuffling, split, and task-domain assignment.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Per-domain-per-class train split ratio.",
    )
    return parser.parse_args()


def iter_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    domains = sorted(path for path in root.iterdir() if path.is_dir())
    if not domains:
        raise ValueError(f"No domains found under: {root}")

    for domain_dir in domains:
        class_dirs = sorted(path for path in domain_dir.iterdir() if path.is_dir())
        if not class_dirs:
            raise ValueError(f"No classes found under domain: {domain_dir}")
        for class_dir in class_dirs:
            image_paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file()
                and not path.name.startswith(".")
                and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not image_paths:
                raise ValueError(f"No images found in: {class_dir}")
            for image_path in image_paths:
                relpath = image_path.relative_to(root).as_posix()
                samples.append(
                    Sample(
                        relpath=relpath,
                        domain=domain_dir.name,
                        class_name=class_dir.name,
                    )
                )
    return samples


def build_task_sizes(num_classes: int, init_cls: int, increment: int) -> list[int]:
    if init_cls <= 0 or increment <= 0:
        raise ValueError("init_cls and increment must be positive.")
    if init_cls > num_classes:
        raise ValueError(
            f"init_cls={init_cls} is larger than the number of classes={num_classes}."
        )

    task_sizes = [init_cls]
    while sum(task_sizes) + increment < num_classes:
        task_sizes.append(increment)
    remainder = num_classes - sum(task_sizes)
    if remainder > 0:
        task_sizes.append(remainder)
    return task_sizes


def split_into_tasks(items: list[str], task_sizes: list[int]) -> list[list[str]]:
    tasks: list[list[str]] = []
    cursor = 0
    for task_size in task_sizes:
        tasks.append(items[cursor : cursor + task_size])
        cursor += task_size
    return tasks


def split_train_test(
    samples: list[Sample], train_ratio: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")

    buckets: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for sample in samples:
        buckets[(sample.domain, sample.class_name)].append(sample)

    train_samples: list[Sample] = []
    test_samples: list[Sample] = []
    for bucket_key in sorted(buckets):
        bucket = buckets[bucket_key]
        rng = random.Random(f"{seed}:{bucket_key[0]}:{bucket_key[1]}")
        bucket = bucket[:]
        rng.shuffle(bucket)
        split_idx = int(len(bucket) * train_ratio)
        split_idx = max(1, min(len(bucket) - 1, split_idx))
        train_samples.extend(bucket[:split_idx])
        test_samples.extend(bucket[split_idx:])

    return train_samples, test_samples


def choose_task_domains(domains: list[str], num_tasks: int, seed: int) -> list[str]:
    shuffled_domains = domains[:]
    random.Random(seed).shuffle(shuffled_domains)
    return [shuffled_domains[idx % len(shuffled_domains)] for idx in range(num_tasks)]


def generate_train_lines(
    train_samples: list[Sample],
    class_to_label: dict[str, int],
    class_to_task_domain: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    for sample in train_samples:
        task_domain = class_to_task_domain.get(sample.class_name)
        if task_domain is None or sample.domain != task_domain:
            continue
        lines.append(f"{sample.relpath} {class_to_label[sample.class_name]}")
    return sorted(lines)


def generate_test_lines(
    test_samples: list[Sample], class_to_label: dict[str, int]
) -> list[str]:
    lines = [
        f"{sample.relpath} {class_to_label[sample.class_name]}"
        for sample in test_samples
        if sample.class_name in class_to_label
    ]
    return sorted(lines)


def main() -> None:
    args = parse_args()
    samples = iter_samples(args.root)

    domains = sorted({sample.domain for sample in samples})
    classes = sorted({sample.class_name for sample in samples})
    if len(classes) != 65:
        raise ValueError(f"Expected 65 classes, found {len(classes)}.")

    train_samples, test_samples = split_train_test(
        samples=samples, train_ratio=args.train_ratio, seed=args.seed
    )

    class_order = classes[:]
    random.Random(args.seed).shuffle(class_order)
    task_sizes = build_task_sizes(len(class_order), args.init_cls, args.increment)
    tasks = split_into_tasks(class_order, task_sizes)
    task_domains = choose_task_domains(domains, len(tasks), args.seed)

    class_to_label = {class_name: idx for idx, class_name in enumerate(classes)}
    class_to_task_domain = {
        class_name: task_domains[task_idx]
        for task_idx, task_classes in enumerate(tasks)
        for class_name in task_classes
    }

    train_lines = generate_train_lines(
        train_samples=train_samples,
        class_to_label=class_to_label,
        class_to_task_domain=class_to_task_domain,
    )
    test_lines = generate_test_lines(
        test_samples=test_samples, class_to_label=class_to_label
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.txt").write_text(
        "\n".join(train_lines) + "\n", encoding="utf-8"
    )
    (args.output_dir / "test.txt").write_text(
        "\n".join(test_lines) + "\n", encoding="utf-8"
    )

    metadata = {
        "root": str(args.root),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "domains": domains,
        "classes": classes,
        "class_order": class_order,
        "task_sizes": task_sizes,
        "task_domains": task_domains,
        "tasks": [
            {
                "task_id": task_idx,
                "domain": task_domains[task_idx],
                "classes": task_classes,
            }
            for task_idx, task_classes in enumerate(tasks)
        ],
        "class_to_label": class_to_label,
        "train_sample_count": len(train_lines),
        "test_sample_count": len(test_lines),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "train_sample_count": len(train_lines),
                "test_sample_count": len(test_lines),
                "task_domains": task_domains,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
