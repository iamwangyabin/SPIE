#!/usr/bin/env python3

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")


@dataclass(frozen=True)
class TxtRef:
    config_path: Path
    config_key: str
    txt_path: Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = repo_root / "utils" / "datautils" / "domainnet" / "cleaned_tuna_lists"
    parser = argparse.ArgumentParser(
        description=(
            "Prune a DomainNet directory based on the txt lists referenced by repo configs, "
            "and emit calibrated txt files with missing entries removed."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="DomainNet root directory, e.g. /path/to/domainnet",
    )
    parser.add_argument(
        "--repo-root",
        default=repo_root,
        type=Path,
        help="SPIE repo root. Defaults to the parent of this script.",
    )
    parser.add_argument(
        "--config-glob",
        default="exps/*domainnet*.json",
        help="Glob pattern, relative to repo root, used to select configs.",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        type=Path,
        help="Directory for calibrated txt files and the summary report.",
    )
    parser.add_argument(
        "--execute-delete",
        action="store_true",
        help="Actually delete files that are not referenced by the selected txt files.",
    )
    parser.add_argument(
        "--drop-empty-dirs",
        action="store_true",
        help="Remove empty class directories after deleting unused files.",
    )
    return parser.parse_args()


def iter_config_paths(repo_root: Path, config_glob: str) -> list[Path]:
    return sorted(repo_root.glob(config_glob))


def collect_txt_refs(repo_root: Path, config_paths: list[Path]) -> tuple[list[TxtRef], list[dict]]:
    txt_refs: list[TxtRef] = []
    missing_txts: list[dict] = []

    for config_path in config_paths:
        config = json.loads(config_path.read_text())
        for key in ("domainnet_train_txt", "domainnet_test_txt"):
            txt_value = config.get(key)
            if not txt_value:
                continue
            txt_path = (repo_root / txt_value).resolve()
            if txt_path.is_file():
                txt_refs.append(TxtRef(config_path=config_path.resolve(), config_key=key, txt_path=txt_path))
            else:
                missing_txts.append(
                    {
                        "config": str(config_path.resolve()),
                        "key": key,
                        "missing_txt": str(txt_path),
                    }
                )

    return txt_refs, missing_txts


def load_txt_lines(txt_path: Path) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    with txt_path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(" ", 1)
            entries.append((rel_path, int(label)))
    return entries


def collect_dataset_files(dataset_root: Path) -> set[str]:
    files: set[str] = set()
    for domain in DOMAINS:
        domain_root = dataset_root / domain
        if not domain_root.is_dir():
            continue
        for path in domain_root.rglob("*"):
            if path.is_file():
                files.add(path.relative_to(dataset_root).as_posix())
    return files


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def relative_txt_output_path(repo_root: Path, txt_path: Path, output_dir: Path) -> Path:
    domainnet_root = repo_root / "utils" / "datautils" / "domainnet"
    try:
        rel = txt_path.relative_to(domainnet_root.resolve())
    except ValueError:
        rel = Path(txt_path.name)
    return output_dir / rel


def write_calibrated_txts(
    repo_root: Path,
    output_dir: Path,
    txt_refs: list[TxtRef],
    existing_files: set[str],
) -> list[dict]:
    written: list[dict] = []
    for txt_ref in txt_refs:
        entries = load_txt_lines(txt_ref.txt_path)
        kept_lines = [f"{rel_path} {label}" for rel_path, label in entries if rel_path in existing_files]
        missing_count = len(entries) - len(kept_lines)
        out_path = relative_txt_output_path(repo_root, txt_ref.txt_path, output_dir)
        ensure_parent(out_path)
        out_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))
        written.append(
            {
                "config": str(txt_ref.config_path),
                "key": txt_ref.config_key,
                "source_txt": str(txt_ref.txt_path),
                "output_txt": str(out_path),
                "original_lines": len(entries),
                "kept_lines": len(kept_lines),
                "dropped_missing_lines": missing_count,
            }
        )
    return written


def prune_files(dataset_root: Path, unused_files: list[str], drop_empty_dirs: bool) -> dict:
    deleted_files = 0
    deleted_dirs = 0

    for rel_path in unused_files:
        target = dataset_root / rel_path
        if target.is_file():
            target.unlink()
            deleted_files += 1

    if drop_empty_dirs:
        for domain in DOMAINS:
            domain_root = dataset_root / domain
            if not domain_root.is_dir():
                continue
            for path in sorted(domain_root.rglob("*"), reverse=True):
                if path.is_dir():
                    try:
                        path.rmdir()
                        deleted_dirs += 1
                    except OSError:
                        pass

    return {"deleted_files": deleted_files, "deleted_dirs": deleted_dirs}


def build_summary(
    config_paths: list[Path],
    txt_refs: list[TxtRef],
    missing_txts: list[dict],
    used_files: set[str],
    existing_files: set[str],
    unused_files: list[str],
    written_txts: list[dict],
    delete_stats: dict | None,
) -> dict:
    return {
        "config_count": len(config_paths),
        "configs": [str(path.resolve()) for path in config_paths],
        "resolved_txt_count": len(txt_refs),
        "missing_txt_count": len(missing_txts),
        "missing_txts": missing_txts,
        "referenced_file_count": len(used_files),
        "existing_referenced_file_count": sum(rel_path in existing_files for rel_path in used_files),
        "missing_referenced_file_count": sum(rel_path not in existing_files for rel_path in used_files),
        "existing_dataset_file_count": len(existing_files),
        "unused_existing_file_count": len(unused_files),
        "calibrated_txts": written_txts,
        "delete_stats": delete_stats,
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()

    config_paths = iter_config_paths(repo_root, args.config_glob)
    if not config_paths:
        raise SystemExit(f"No configs matched: {args.config_glob}")

    txt_refs, missing_txts = collect_txt_refs(repo_root, config_paths)
    if not txt_refs:
        raise SystemExit("No existing DomainNet txt files were found in the selected configs.")

    used_files: set[str] = set()
    for txt_ref in txt_refs:
        for rel_path, _ in load_txt_lines(txt_ref.txt_path):
            used_files.add(rel_path)

    existing_files = collect_dataset_files(dataset_root)
    unused_files = sorted(existing_files - used_files)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_txts = write_calibrated_txts(repo_root, output_dir, txt_refs, existing_files)

    delete_stats = None
    if args.execute_delete:
        delete_stats = prune_files(dataset_root, unused_files, args.drop_empty_dirs)

    summary = build_summary(
        config_paths=config_paths,
        txt_refs=txt_refs,
        missing_txts=missing_txts,
        used_files=used_files,
        existing_files=existing_files,
        unused_files=unused_files,
        written_txts=written_txts,
        delete_stats=delete_stats,
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"Configs matched: {summary['config_count']}")
    print(f"Existing txt refs: {summary['resolved_txt_count']}")
    print(f"Missing txt refs: {summary['missing_txt_count']}")
    print(f"Referenced files: {summary['referenced_file_count']}")
    print(f"Missing referenced files: {summary['missing_referenced_file_count']}")
    print(f"Unused existing files: {summary['unused_existing_file_count']}")
    print(f"Calibrated txt output: {output_dir}")
    print(f"Summary report: {summary_path}")
    if delete_stats is not None:
        print(f"Deleted files: {delete_stats['deleted_files']}")
        print(f"Deleted empty dirs: {delete_stats['deleted_dirs']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
