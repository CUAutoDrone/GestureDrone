# Reads raw landmark CSV from GestureData_Command4.py
# Applies:
# - wrist-centering
# - scale normalization
# - mirroring left hand to canonical right-hand layout
# - bone / knuckle difference vectors
# - bone lengths
# - palm spread distances
# - fingertip-to-wrist distances
#
# Writes processed features to:
# gesture_data_command4_processed.csv

import csv
import os
import numpy as np

RAW_CSV = "gesture_data_command4_raw.csv"
PROCESSED_CSV = "gesture_data_command4_processed.csv"

# Landmark indices (MediaPipe Hands)
WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Adjacent joints / "knuckle differences"
BONES = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
]

def safe_norm(v):
    return np.linalg.norm(v) + 1e-8

def normalize_landmarks(landmarks, handedness):
    """
    landmarks: (21,3) raw mediapipe coords
    returns: canonical normalized landmarks (21,3)
    """
    pts = landmarks.copy()

    # 1) Center at wrist
    wrist = pts[WRIST].copy()
    pts = pts - wrist

    # 2) Scale normalize by palm size
    # use avg distance wrist -> index_mcp/middle_mcp/pinky_mcp
    ref_points = [INDEX_MCP, MIDDLE_MCP, PINKY_MCP]
    scale = np.mean([safe_norm(pts[i]) for i in ref_points])

    if scale < 1e-8:
        scale = 1.0

    pts = pts / scale

    # 3) Mirror left hand so both hands map to same canonical orientation
    if handedness == "Left":
        pts[:, 0] *= -1.0

    return pts

def extract_features(norm_pts):
    """
    norm_pts: (21,3)
    returns: 1D feature vector
    """
    features = []

    # A) normalized coordinates themselves
    features.extend(norm_pts.flatten().tolist())

    # B) adjacent joint / bone difference vectors
    #    this is the "difference between knuckles" idea
    bone_lengths = []
    for a, b in BONES:
        vec = norm_pts[b] - norm_pts[a]
        features.extend(vec.tolist())
        bone_lengths.append(np.linalg.norm(vec))

    # C) bone lengths
    features.extend(bone_lengths)

    # D) palm spread distances between MCP knuckles
    palm_spread = [
        np.linalg.norm(norm_pts[INDEX_MCP] - norm_pts[MIDDLE_MCP]),
        np.linalg.norm(norm_pts[MIDDLE_MCP] - norm_pts[RING_MCP]),
        np.linalg.norm(norm_pts[RING_MCP] - norm_pts[PINKY_MCP]),
    ]
    features.extend(palm_spread)

    # E) fingertip to wrist distances
    tip_to_wrist = [np.linalg.norm(norm_pts[t]) for t in TIPS]
    features.extend(tip_to_wrist)

    # F) fingertip direction vectors relative to MCP
    finger_dirs = [
        norm_pts[THUMB_TIP] - norm_pts[THUMB_MCP],
        norm_pts[INDEX_TIP] - norm_pts[INDEX_MCP],
        norm_pts[MIDDLE_TIP] - norm_pts[MIDDLE_MCP],
        norm_pts[RING_TIP] - norm_pts[RING_MCP],
        norm_pts[PINKY_TIP] - norm_pts[PINKY_MCP],
    ]
    for vec in finger_dirs:
        features.extend(vec.tolist())

    return features

def main():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Could not find {RAW_CSV}")

    processed_rows = []
    header_written = False
    output_header = None

    with open(RAW_CSV, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            raw = np.array([float(row[f"f{i}"]) for i in range(63)], dtype=np.float32).reshape(21, 3)
            handedness = row["handedness"]
            label = row["label"]

            norm_pts = normalize_landmarks(raw, handedness)
            feats = extract_features(norm_pts)

            if not header_written:
                output_header = [f"feat_{i}" for i in range(len(feats))] + ["label"]
                header_written = True

            processed_rows.append(feats + [label])

    with open(PROCESSED_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(output_header)
        writer.writerows(processed_rows)

    print(f"Processed {len(processed_rows)} rows.")
    print(f"Saved processed data to {PROCESSED_CSV}")

if __name__ == "__main__":
    main()