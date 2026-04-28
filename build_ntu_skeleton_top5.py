#!/usr/bin/env python3
"""Curate NTU 120 skeletons into a 5-class, common-subject subset.

Default behaviour selects the lexicographically smallest valid candidate
per (subject, class), which collapses cameras/replications to mostly
C001/R001. Pass --seed N to instead pick a deterministically varied
candidate per pair, spreading the selection across all available
(camera, replication) combinations. Same 69 subjects × 5 classes = 345
samples are produced either way.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data"
SOURCE_DIRS = [
    DATA_ROOT / "nturgbd_skeletons_s001_to_s017" / "nturgb+d_skeletons",
    DATA_ROOT / "nturgbd_skeletons_s018_to_s032",
]
MISSING_LISTS = [
    DATA_ROOT / "ntu_rgbd120_missing_incomplete_skeletons_s001_to_s017.txt",
    DATA_ROOT / "ntu_rgbd120_missing_incomplete_skeletons_s018_to_s032.txt",
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
        if not list_path.exists():
            continue
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

            # Missing-incomplete txt lists paths relative to DATA_ROOT
            # (e.g. "nturgbd_skeletons_s018_to_s032/S019C001P046R001A075.skeleton").
            relative_path = path.relative_to(DATA_ROOT).as_posix()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, deterministically pick a varied (camera, replication) "
             "candidate per (subject, class) instead of the sorted-first one. "
             "Selection is per-pair: the candidate list is shuffled with "
             "seed=(args.seed, person_id, class_id) and the first element "
             "is taken — same input -> same output, but the choice spreads "
             "across all available (camera, replication) combinations.",
    )
    return parser.parse_args()


def select_candidate(paths: list[Path], person_id: str, class_id: str,
                     seed: int | None) -> Path:
    if seed is None:
        return paths[0]
    rng = random.Random((seed, person_id, class_id))
    perm = list(paths)
    rng.shuffle(perm)
    return perm[0]


def main() -> int:
    args = parse_args()
    candidates, subjects_by_class = collect_valid_candidates()
    common_subjects = sorted(set.intersection(*(subjects_by_class[class_id] for class_id in TARGET_CLASSES)))

    reset_output_dir()

    manifest_rows: list[dict[str, str]] = []

    for class_id, class_name in TARGET_CLASSES.items():
        class_dir = OUTPUT_DIR / f"{class_id}_{class_name}"
        class_dir.mkdir()

        for person_id in common_subjects:
            selected_path = select_candidate(
                candidates[(person_id, class_id)], person_id, class_id, args.seed
            )
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
