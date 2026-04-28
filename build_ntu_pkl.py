#!/usr/bin/env python3
"""Build per-activity skeleton pickles from ntu_skeleton/ in the same
format as data/data_stroke.pkl: dict[person_id (str) -> ndarray of shape
(num_joints=25, 3, num_frames)].
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
NTU_DIR = REPO_ROOT / "ntu_skeleton"
NUM_JOINTS = 25


def parse_skeleton(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as fh:
        frame_count = int(fh.readline().strip())
        frames = np.empty((NUM_JOINTS, 3, frame_count), dtype=np.float32)

        for f in range(frame_count):
            body_count = int(fh.readline().strip())
            assert body_count == 1, f"{path} frame {f} has {body_count} bodies"
            fh.readline()  # body metadata
            joint_count = int(fh.readline().strip())
            assert joint_count == NUM_JOINTS

            for j in range(NUM_JOINTS):
                parts = fh.readline().split()
                frames[j, 0, f] = float(parts[0])
                frames[j, 1, f] = float(parts[1])
                frames[j, 2, f] = float(parts[2])

    return frames


def person_id(path: Path) -> str:
    # filename like S018C001P008R001A080.skeleton; person id = digits after P
    return str(int(path.stem[9:12]))


def main() -> int:
    class_dirs = sorted(p for p in NTU_DIR.iterdir() if p.is_dir() and p.name.startswith("A"))

    combined: dict[str, np.ndarray] = {}

    for class_dir in class_dirs:
        class_name = class_dir.name  # e.g. A080_squat_down
        class_id = class_name.split("_", 1)[0]  # e.g. A080
        per_subject: dict[str, np.ndarray] = {}
        for skel_path in sorted(class_dir.glob("*.skeleton")):
            pid = person_id(skel_path)
            arr = parse_skeleton(skel_path)
            per_subject[pid] = arr
            combined[f"{pid}_{class_id}"] = arr

        out_path = NTU_DIR / f"data_{class_name}.pkl"
        with out_path.open("wb") as fh:
            pickle.dump(per_subject, fh, protocol=pickle.HIGHEST_PROTOCOL)

        sample_pid = next(iter(per_subject))
        print(f"{class_name}: {len(per_subject)} subjects, sample shape {per_subject[sample_pid].shape} -> {out_path.relative_to(REPO_ROOT)}")

    combined_path = REPO_ROOT / "data" / "data_ntu.pkl"
    with combined_path.open("wb") as fh:
        pickle.dump(combined, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"combined -> {combined_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
