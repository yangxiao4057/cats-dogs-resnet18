"""检查原始图片，按像素去重，并保存可复现的猫狗数据划分。"""

import argparse
import csv
import hashlib
import json
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


CLASS_TO_IDX = {"cat": 0, "dog": 1}
RAW_FOLDERS = {"cat": "Cat", "dog": "Dog"}
# 先复现原始 3000 张划分，再缩减训练部分，保证验证、测试图片不变。
BASE_SPLIT_COUNTS = {"train": 1050, "val": 225, "test": 225}
SPLIT_COUNTS = {"train": 500, "val": 225, "test": 225}


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(project, seed):
    raw = project / "data" / "raw" / "PetImages"
    output = project / "data" / "splits"
    for folder in RAW_FOLDERS.values():
        if not (raw / folder).is_dir():
            raise FileNotFoundError(f"缺少目录：{raw / folder}")

    groups = defaultdict(list)
    invalid, warning_rows = [], []
    scanned = Counter()
    for category, folder in RAW_FOLDERS.items():
        paths = sorted((raw / folder).glob("*.jpg"))
        for index, path in enumerate(paths, start=1):
            relative = path.relative_to(project).as_posix()
            scanned[category] += 1
            try:
                # verify 检查文件结构；重新打开并 load，保证像素也能完整解码。
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        image.load()
                        rgb = ImageOps.exif_transpose(image).convert("RGB")
                        digest = hashlib.sha256()
                        digest.update(f"{rgb.width}x{rgb.height}:RGB:".encode())
                        digest.update(rgb.tobytes())
                for warning in caught:
                    warning_rows.append({"path": relative, "message": str(warning.message)})
                if any("truncated" in str(w.message).lower() or "corrupt" in str(w.message).lower() for w in caught):
                    raise ValueError("解码产生截断或损坏警告，保守排除")
                groups[digest.hexdigest()].append({
                    "path": relative,
                    "class_name": category,
                    "label": CLASS_TO_IDX[category],
                    "pixel_sha256": digest.hexdigest(),
                })
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                invalid.append({"path": relative, "reason": f"{type(error).__name__}: {error}"})
            if index % 2500 == 0 or index == len(paths):
                print(f"检查 {category}: {index}/{len(paths)}", flush=True)

    unique = {category: [] for category in CLASS_TO_IDX}
    duplicates, conflicts = [], []
    for digest, members in sorted(groups.items()):
        members.sort(key=lambda row: row["path"])
        if len({row["label"] for row in members}) > 1:
            # 相同像素出现猫、狗两种标签时，整个组都不参与实验。
            conflicts.extend(members)
            continue
        kept, *removed = members
        unique[kept["class_name"]].append(kept)
        for row in removed:
            duplicates.append({**row, "kept_path": kept["path"]})

    splits = {name: [] for name in SPLIT_COUNTS}
    rng = random.Random(seed)
    for category in CLASS_TO_IDX:
        candidates = sorted(unique[category], key=lambda row: row["path"])
        required = sum(BASE_SPLIT_COUNTS.values())
        if len(candidates) < required:
            raise RuntimeError(f"{category} 有效独立图片不足 {required} 张")
        selected = rng.sample(candidates, required)
        start = 0
        for name, original_count in BASE_SPLIT_COUNTS.items():
            splits[name].extend(selected[start:start + SPLIT_COUNTS[name]])
            start += original_count

    # 验证数量、类别平衡，以及三个集合没有路径或完全相同的像素交叉。
    all_rows = []
    for name, rows in splits.items():
        if Counter(row["class_name"] for row in rows) != {
            category: SPLIT_COUNTS[name] for category in CLASS_TO_IDX
        }:
            raise RuntimeError(f"划分数量异常：{name}")
        all_rows.extend(rows)
    expected_total = len(CLASS_TO_IDX) * sum(SPLIT_COUNTS.values())
    if len({row["path"] for row in all_rows}) != expected_total:
        raise RuntimeError("划分存在路径重复")
    if len({row["pixel_sha256"] for row in all_rows}) != expected_total:
        raise RuntimeError("划分存在像素重复")

    output.mkdir(parents=True, exist_ok=True)
    fields = ["path", "class_name", "label", "pixel_sha256"]
    for name, rows in splits.items():
        write_csv(output / f"{name}.csv", rows, fields)
    write_csv(output / "invalid_images.csv", invalid, ["path", "reason"])
    write_csv(output / "duplicate_images.csv", duplicates, fields + ["kept_path"])
    write_csv(output / "label_conflicts.csv", conflicts, fields)
    write_csv(output / "image_warnings.csv", warning_rows, ["path", "message"])
    summary = {
        "seed": seed,
        "selection_method": "Reproduce original 1500/class sample; retain first 500/class from original training allocation; validation/test unchanged",
        "class_to_idx": CLASS_TO_IDX,
        "source": "https://www.microsoft.com/en-us/download/details.aspx?id=54765",
        "scanned_per_class": dict(scanned),
        "invalid_images": len(invalid),
        "warning_records": len(warning_rows),
        "same_class_duplicates_removed": len(duplicates),
        "cross_class_conflict_images_removed": len(conflicts),
        "eligible_unique_per_class": {key: len(value) for key, value in unique.items()},
        "splits": {
            name: {"total": len(rows), "per_class": dict(Counter(row["class_name"] for row in rows))}
            for name, rows in splits.items()
        },
        "duplicate_method": "SHA256 of EXIF-oriented decoded RGB pixels and dimensions; near-duplicates not checked",
        "raw_images_modified": False,
    }
    for name, content in (("class_to_idx.json", CLASS_TO_IDX), ("summary.json", summary)):
        (output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"清单已保存：{output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.project_dir.resolve(), args.seed)
