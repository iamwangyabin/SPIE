#!/usr/bin/env python3
"""Generate a strong DomainNet class-incremental protocol.

Protocol:
1. Select the top-k classes ranked by source train image count.
2. Shuffle the selected classes with a fixed seed.
3. Split the shuffled classes into incremental tasks by init_cls/increment.
4. Assign exactly one training domain to each task.
5. Build:
   - train.txt: for each task, only the assigned domain samples of the task classes
   - test.txt: for all selected classes, keep all available test-domain samples

The script expects source DomainNet train/test txt files in the standard format:
    relative/path/to/image.jpg label

Class names and domains are inferred from the relative path:
    <domain>/<class_name>/<image_name>
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Sample:
    relpath: str
    domain: str
    class_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a strong DomainNet train/test split."
    )
    parser.add_argument(
        "--source-train-txt",
        type=Path,
        nargs="+",
        required=True,
        help="One or more source DomainNet training txt files from the original/full split.",
    )
    parser.add_argument(
        "--source-test-txt",
        type=Path,
        nargs="+",
        required=True,
        help="One or more source DomainNet test txt files from the original/full split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write generated train.txt/test.txt/metadata.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Select the top-k classes by source train image count.",
    )
    parser.add_argument(
        "--init-cls",
        type=int,
        default=20,
        help="Number of classes in the first task.",
    )
    parser.add_argument(
        "--increment",
        type=int,
        default=20,
        help="Number of classes in each following task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1993,
        help="Random seed for class shuffling and task-domain assignment.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5000,
        help="Maximum retries to find a feasible class order and task-domain schedule.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help=(
            "Optional domain whitelist. If omitted, domains are inferred from the "
            "source train split."
        ),
    )
    return parser.parse_args()


def load_samples(
    txt_paths: Iterable[Path], allowed_domains: set[str] | None = None
) -> list[Sample]:
    samples: list[Sample] = []
    txt_paths = list(txt_paths)
    for txt_path in txt_paths:
        with txt_path.open("r", encoding="utf-8") as handle:
            for lineno, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    relpath, _ = line.split(" ")
                    domain, class_name, _ = relpath.split("/", 2)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid line format in {txt_path} at line {lineno}: {line}"
                    ) from exc
                if allowed_domains is not None and domain not in allowed_domains:
                    continue
                samples.append(
                    Sample(relpath=relpath, domain=domain, class_name=class_name)
                )
    if not samples:
        joined_paths = ", ".join(str(path) for path in txt_paths)
        raise ValueError(f"No usable samples were found in: {joined_paths}")
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


def select_top_classes(train_samples: Iterable[Sample], top_k: int) -> list[str]:
    train_counts = Counter(sample.class_name for sample in train_samples)
    if top_k > len(train_counts):
        raise ValueError(
            f"Requested top_k={top_k}, but only {len(train_counts)} classes exist."
        )
    ranked = sorted(train_counts.items(), key=lambda item: (-item[1], item[0]))
    return [class_name for class_name, _ in ranked[:top_k]]


def build_domain_index(samples: Iterable[Sample]) -> dict[str, set[str]]:
    class_to_domains: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        class_to_domains[sample.class_name].add(sample.domain)
    return class_to_domains


def choose_task_domains(
    selected_classes: list[str],
    task_sizes: list[int],
    class_train_domains: dict[str, set[str]],
    domains: list[str],
    seed: int,
    max_attempts: int,
) -> tuple[list[str], list[list[str]], list[str]]:
    base_domains = list(domains)
    if not base_domains:
        raise ValueError("No domains are available for task assignment.")

    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        class_order = selected_classes[:]
        rng.shuffle(class_order)
        tasks = split_into_tasks(class_order, task_sizes)

        shuffled_domains = base_domains[:]
        rng.shuffle(shuffled_domains)

        task_domains: list[str] = []
        feasible = True
        for task_idx, task_classes in enumerate(tasks):
            common_domains = set(shuffled_domains)
            for class_name in task_classes:
                common_domains &= class_train_domains[class_name]
            if not common_domains:
                feasible = False
                break

            preferred_order = shuffled_domains[task_idx % len(shuffled_domains) :] + shuffled_domains[
                : task_idx % len(shuffled_domains)
            ]
            task_domain = next(
                domain for domain in preferred_order if domain in common_domains
            )
            task_domains.append(task_domain)

        if feasible:
            return class_order, tasks, task_domains

    raise RuntimeError(
        "Failed to find a feasible task-domain schedule. Try a different seed, "
        "smaller task size, or a different domain whitelist."
    )


def generate_train_lines(
    train_samples: Iterable[Sample],
    class_to_label: dict[str, int],
    class_to_task_domain: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    for sample in train_samples:
        task_domain = class_to_task_domain.get(sample.class_name)
        if task_domain is None or sample.domain != task_domain:
            continue
        lines.append(f"{sample.relpath} {class_to_label[sample.class_name]}")
    return lines


def generate_test_lines(
    test_samples: Iterable[Sample],
    class_to_label: dict[str, int],
) -> list[str]:
    lines: list[str] = []
    for sample in test_samples:
        label = class_to_label.get(sample.class_name)
        if label is None:
            continue
        lines.append(f"{sample.relpath} {label}")
    return lines


def ensure_test_coverage(
    class_order: list[str], test_samples: Iterable[Sample]
) -> dict[str, dict[str, int]]:
    test_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sample in test_samples:
        if sample.class_name in class_order:
            test_counts[sample.class_name][sample.domain] += 1

    missing_classes = [class_name for class_name in class_order if class_name not in test_counts]
    if missing_classes:
        preview = ", ".join(missing_classes[:10])
        raise ValueError(
            f"Some selected classes do not exist in the source test split: {preview}"
        )

    return {class_name: dict(domain_counts) for class_name, domain_counts in test_counts.items()}


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def main() -> None:
    args = parse_args()

    allowed_domains = set(args.domains) if args.domains else None
    train_samples = load_samples(args.source_train_txt, allowed_domains=allowed_domains)
    inferred_domains = sorted({sample.domain for sample in train_samples})
    domains = args.domains if args.domains else inferred_domains

    test_samples = load_samples(args.source_test_txt, allowed_domains=set(domains))
    selected_classes = select_top_classes(train_samples, args.top_k)
    task_sizes = build_task_sizes(len(selected_classes), args.init_cls, args.increment)
    class_train_domains = build_domain_index(
        sample for sample in train_samples if sample.class_name in set(selected_classes)
    )

    class_order, tasks, task_domains = choose_task_domains(
        selected_classes=selected_classes,
        task_sizes=task_sizes,
        class_train_domains=class_train_domains,
        domains=domains,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )

    class_to_label = {class_name: label for label, class_name in enumerate(class_order)}
    class_to_task_domain = {
        class_name: task_domain
        for task_classes, task_domain in zip(tasks, task_domains)
        for class_name in task_classes
    }

    train_lines = generate_train_lines(
        train_samples=train_samples,
        class_to_label=class_to_label,
        class_to_task_domain=class_to_task_domain,
    )
    test_lines = generate_test_lines(
        test_samples=test_samples,
        class_to_label=class_to_label,
    )

    if not train_lines:
        raise ValueError("Generated train split is empty.")
    if not test_lines:
        raise ValueError("Generated test split is empty.")

    selected_train_counts = Counter()
    for line in train_lines:
        _, label_str = line.rsplit(" ", 1)
        selected_train_counts[int(label_str)] += 1
    missing_train_labels = [
        label for label in range(len(class_order)) if selected_train_counts[label] == 0
    ]
    if missing_train_labels:
        raise ValueError(
            f"Some selected classes have no training samples after filtering: {missing_train_labels[:10]}"
        )

    test_coverage = ensure_test_coverage(class_order, test_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / "train.txt"
    test_out = args.output_dir / "test.txt"
    metadata_out = args.output_dir / "metadata.json"

    write_lines(train_out, train_lines)
    write_lines(test_out, test_lines)

    metadata = {
        "seed": args.seed,
        "source_train_txts": [str(path) for path in args.source_train_txt],
        "source_test_txts": [str(path) for path in args.source_test_txt],
        "top_k": args.top_k,
        "init_cls": args.init_cls,
        "increment": args.increment,
        "domains": domains,
        "task_sizes": task_sizes,
        "task_domains": task_domains,
        "tasks": tasks,
        "class_order": class_order,
        "class_to_label": class_to_label,
        "class_to_task_domain": class_to_task_domain,
        "test_coverage": test_coverage,
        "num_train_samples": len(train_lines),
        "num_test_samples": len(test_lines),
    }
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {train_out}")
    print(f"Wrote {test_out}")
    print(f"Wrote {metadata_out}")
    print(f"Selected {len(class_order)} classes across {len(tasks)} tasks.")
    print(f"Task domains: {task_domains}")


if __name__ == "__main__":
    main()
