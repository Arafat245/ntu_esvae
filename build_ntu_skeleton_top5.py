#!/usr/bin/env python3

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIRS = [
    REPO_ROOT / "nturgbd_skeletons_s001_to_s017" / "nturgb+d_skeletons",
    REPO_ROOT / "nturgbd_skeletons_s018_to_s032",
]
MISSING_LISTS = [
    REPO_ROOT / "ntu_rgbd120_missing_incomplete_skeletons_s001_to_s017.txt",
    REPO_ROOT / "ntu_rgbd120_missing_incomplete_skeletons_s018_to_s032.txt",
]
OUTPUT_DIR = REPO_ROOT / "ntu_skeleton"

# Five one-person classes chosen for maximum usable subject coverage under the
# single-skeleton and missing-sample filtering rule.
TARGET_CLASSES = {
    "A080": "squat_down",
    "A097": "arm_circles",
    "A098": "arm_swings",
    "A100": "kick_backward",
    "A101": "cross_toe_touch",
}


def load_missing_paths() -> set[str]:
    missing_paths: set[str] = set()
    for list_path in MISSING_LISTS:
        for raw_line in list_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line:
                missing_paths.add(line)
    return missing_paths


def extract_class_id(path: Path) -> str:
    return path.stem[16:20]


def extract_person_id(path: Path) -> str:
    return path.stem[8:12]


def is_single_skeleton_trial(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        try:
            frame_count = int(handle.readline().strip())
        except ValueError:
            return False

        for _ in range(frame_count):
            body_count_line = handle.readline()
            if not body_count_line:
                return False

            try:
                body_count = int(body_count_line.strip())
            except ValueError:
                return False

            if body_count != 1:
                return False

            # Body metadata line.
            if not handle.readline():
                return False

            joint_count_line = handle.readline()
            if not joint_count_line:
                return False

            try:
                joint_count = int(joint_count_line.strip())
            except ValueError:
                return False

            for _ in range(joint_count):
                if not handle.readline():
                    return False

    return True


def collect_valid_candidates() -> tuple[dict[tuple[str, str], list[Path]], dict[str, set[str]]]:
    missing_paths = load_missing_paths()
    candidates: dict[tuple[str, str], list[Path]] = defaultdict(list)
    subjects_by_class: dict[str, set[str]] = defaultdict(set)

    for source_dir in SOURCE_DIRS:
        for path in sorted(source_dir.glob("*.skeleton")):
            class_id = extract_class_id(path)
            if class_id not in TARGET_CLASSES:
                continue

            relative_path = path.relative_to(REPO_ROOT).as_posix()
            if relative_path in missing_paths:
                continue

            if not is_single_skeleton_trial(path):
                continue

            person_id = extract_person_id(path)
            candidates[(person_id, class_id)].append(path)
            subjects_by_class[class_id].add(person_id)

    return candidates, subjects_by_class


def reset_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()


def main() -> int:
    candidates, subjects_by_class = collect_valid_candidates()
    common_subjects = sorted(set.intersection(*(subjects_by_class[class_id] for class_id in TARGET_CLASSES)))

    reset_output_dir()

    manifest_rows: list[dict[str, str]] = []

    for class_id, class_name in TARGET_CLASSES.items():
        class_dir = OUTPUT_DIR / f"{class_id}_{class_name}"
        class_dir.mkdir()

        for person_id in common_subjects:
            selected_path = candidates[(person_id, class_id)][0]
            output_path = class_dir / selected_path.name
            shutil.copy2(selected_path, output_path)

            manifest_rows.append(
                {
                    "person_id": person_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "source_file": selected_path.relative_to(REPO_ROOT).as_posix(),
                    "output_file": output_path.relative_to(REPO_ROOT).as_posix(),
                }
            )

    manifest_path = OUTPUT_DIR / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["person_id", "class_id", "class_name", "source_file", "output_file"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    common_subjects_path = OUTPUT_DIR / "common_subjects.txt"
    common_subjects_path.write_text("\n".join(common_subjects) + "\n", encoding="utf-8")

    print(f"common_subjects={len(common_subjects)}")
    print(f"copied_files={len(manifest_rows)}")
    for class_id, class_name in TARGET_CLASSES.items():
        print(f"{class_id} {class_name} files={len(common_subjects)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
