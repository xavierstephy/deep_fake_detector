import io
import os
import sys
import math
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2


# -------------------------------
# Face Detection
# -------------------------------
def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")
    return detector


def detect_faces_bboxes(image_bgr: np.ndarray, detector: cv2.CascadeClassifier) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


# -------------------------------
# Feature Extraction Heuristics
# -------------------------------
def compute_laplacian_variance(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_high_freq_energy_ratio(gray: np.ndarray) -> float:
    # FFT-based high-frequency energy ratio
    # Normalize and center
    g = gray.astype(np.float32) / 255.0
    f = np.fft.fft2(g)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    # Radius threshold to separate low/high frequency
    r = int(min(h, w) * 0.08)
    Y, X = np.ogrid[:h, :w]
    mask_low = (Y - cy) ** 2 + (X - cx) ** 2 <= r * r
    low = magnitude[mask_low].sum()
    high = magnitude[~mask_low].sum()
    total = low + high + 1e-8
    return float(high / total)


def compute_color_channel_correlation(face_bgr: np.ndarray) -> float:
    # Correlation between channels; manipulated faces often show abnormal correlations
    b, g, r = cv2.split(face_bgr)
    b = b.astype(np.float32).flatten()
    g = g.astype(np.float32).flatten()
    r = r.astype(np.float32).flatten()
    corr_bg = np.corrcoef(b, g)[0, 1]
    corr_br = np.corrcoef(b, r)[0, 1]
    corr_gr = np.corrcoef(g, r)[0, 1]
    corr = np.nanmean([corr_bg, corr_br, corr_gr])
    if np.isnan(corr):
        corr = 0.0
    return float(corr)


def compute_block_dct_consistency(gray: np.ndarray, block_size: int = 8) -> float:
    # Compute variance of DC and a few AC coefficients across blocks as a proxy for stitching/compression artifacts
    h, w = gray.shape
    h8 = (h // block_size) * block_size
    w8 = (w // block_size) * block_size
    gray = gray[:h8, :w8]
    gray = gray.astype(np.float32) - 128.0

    dc_values = []
    ac_values = []
    for y in range(0, h8, block_size):
        for x in range(0, w8, block_size):
            block = gray[y:y + block_size, x:x + block_size]
            dct = cv2.dct(block)
            dc_values.append(dct[0, 0])
            ac_values.append(np.mean(np.abs(dct[0:3, 0:3])))

    dc_var = float(np.var(dc_values)) if dc_values else 0.0
    ac_var = float(np.var(ac_values)) if ac_values else 0.0
    # Higher inconsistency can be indicative of manipulation/compression oddities
    return float(ac_var / (dc_var + 1e-6))


def extract_face_features(face_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    # Standardize size for more stable metrics
    gray_resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    face_resized = cv2.resize(face_bgr, (128, 128), interpolation=cv2.INTER_AREA)

    lap_var = compute_laplacian_variance(gray_resized)
    hf_ratio = compute_high_freq_energy_ratio(gray_resized)
    channel_corr = compute_color_channel_correlation(face_resized)
    dct_consistency = compute_block_dct_consistency(gray_resized)

    return {
        "laplacian_var": lap_var,
        "high_freq_ratio": hf_ratio,
        "channel_corr": channel_corr,
        "dct_consistency": dct_consistency,
    }


# -------------------------------
# Scoring
# -------------------------------
def normalize_feature(value: float, low: float, high: float, invert: bool = False) -> float:
    v = (value - low) / (high - low + 1e-8)
    v = max(0.0, min(1.0, v))
    return 1.0 - v if invert else v


def compute_fake_score(features: Dict[str, float]) -> float:
    # Heuristic normalization ranges derived from typical natural image stats
    # These are not universal. Adjust for your data.
    lap_n = normalize_feature(features["laplacian_var"], low=20.0, high=300.0, invert=True)  # overly smooth → suspicious
    hf_n = normalize_feature(features["high_freq_ratio"], low=0.60, high=0.90, invert=True)  # too few highs → suspicious
    corr_n = normalize_feature(features["channel_corr"], low=0.80, high=0.99, invert=True)   # too high corr → suspicious
    dct_n = normalize_feature(features["dct_consistency"], low=0.10, high=1.50, invert=False) # very inconsistent blocks → suspicious

    # Weighted sum
    weights = {
        "lap": 0.30,
        "hf": 0.30,
        "corr": 0.20,
        "dct": 0.20,
    }
    score = (
        weights["lap"] * lap_n +
        weights["hf"] * hf_n +
        weights["corr"] * corr_n +
        weights["dct"] * dct_n
    )
    # Clamp to [0,1]
    return float(max(0.0, min(1.0, score)))


# -------------------------------
# Image / Video IO
# -------------------------------
def read_image_from_bytes(file_bytes: bytes) -> Optional[np.ndarray]:
    data = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def sample_video_frames(video_path: str, max_frames: int = 64, step: int = 10) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


def draw_face_bboxes(image_bgr: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    vis = image_bgr.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis


def analyze_faces_in_image(image_bgr: np.ndarray, detector: cv2.CascadeClassifier) -> Tuple[List[Dict[str, float]], List[Tuple[int, int, int, int]]]:
    faces = detect_faces_bboxes(image_bgr, detector)
    features_list: List[Dict[str, float]] = []
    for (x, y, w, h) in faces:
        pad = int(0.05 * max(w, h))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image_bgr.shape[1], x + w + pad)
        y1 = min(image_bgr.shape[0], y + h + pad)
        face = image_bgr[y0:y1, x0:x1]
        if face.size == 0:
            continue
        feats = extract_face_features(face)
        features_list.append(feats)
    return features_list, faces


def aggregate_scores(scores: List[float]) -> float:
    if not scores:
        return 0.0
    # Robust aggregation: mean of top-k (k = half, at least 1)
    k = max(1, len(scores) // 2)
    topk = sorted(scores, reverse=True)[:k]
    return float(np.mean(topk))


# -------------------------------
# Streamlit UI
# -------------------------------
def run_streamlit_app():
    try:
        import streamlit as st
    except Exception as exc:
        print("Streamlit is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        raise

    st.set_page_config(page_title="Deepfake Detector (Heuristic Demo)", layout="wide")
    st.title("Deepfake Detector (Heuristic Demo)")
    st.write("This demo uses classical, heuristic features. For production, use a trained model.")

    detector = load_face_detector()

    uploaded_file = st.file_uploader("Upload an image or video", type=[
        "jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff",
        "mp4", "mov", "avi", "mkv", "webm"
    ])

    col_left, col_right = st.columns([1, 1])

    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        is_video = any(filename.endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"])

        if not is_video:
            # Image path
            file_bytes = uploaded_file.read()
            image_bgr = read_image_from_bytes(file_bytes)
            if image_bgr is None:
                st.error("Failed to read image file.")
                return

            with st.spinner("Detecting faces and analyzing..."):
                feats_list, faces = analyze_faces_in_image(image_bgr, detector)
                scores = [compute_fake_score(f) for f in feats_list]
                overall = aggregate_scores(scores)

            vis = draw_face_bboxes(image_bgr, faces)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

            with col_left:
                st.subheader("Input with Detected Faces")
                st.image(vis_rgb, channels="RGB")

            with col_right:
                st.subheader("Results")
                if not feats_list:
                    st.warning("No faces detected.")
                else:
                    for idx, (feats, s) in enumerate(zip(feats_list, scores), start=1):
                        st.write(f"Face {idx} — Fake score: {s:.2f}")
                        st.json(feats)
                st.markdown(f"**Overall fake score**: {overall:.2f}")
                st.progress(min(1.0, overall))

        else:
            # Video path: save to a temp file to read via OpenCV
            tmp_path = os.path.join(os.getcwd(), "_uploaded_video.tmp")
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())

            with st.spinner("Sampling video frames, detecting faces, and analyzing..."):
                frames = sample_video_frames(tmp_path, max_frames=64, step=10)
                per_frame_scores = []
                detector = load_face_detector()
                example_vis = None
                for frame in frames:
                    feats_list, faces = analyze_faces_in_image(frame, detector)
                    scores = [compute_fake_score(f) for f in feats_list]
                    if scores:
                        per_frame_scores.append(float(np.mean(scores)))
                    if example_vis is None:
                        example_vis = draw_face_bboxes(frame, faces)

                overall = aggregate_scores(per_frame_scores)

            if example_vis is not None:
                with col_left:
                    st.subheader("Example Frame with Detected Faces")
                    st.image(cv2.cvtColor(example_vis, cv2.COLOR_BGR2RGB), channels="RGB")

            with col_right:
                st.subheader("Results")
                if not per_frame_scores:
                    st.warning("No faces detected in sampled frames.")
                else:
                    st.line_chart(per_frame_scores)
                st.markdown(f"**Overall fake score**: {overall:.2f}")
                st.progress(min(1.0, overall))

            try:
                os.remove(tmp_path)
            except Exception:
                pass


# -------------------------------
# CLI fallback
# -------------------------------
def run_cli():
    parser = argparse.ArgumentParser(description="Deepfake heuristic detector")
    parser.add_argument("path", help="Path to image or video")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--step", type=int, default=10, help="Frame sampling step for video")
    args = parser.parse_args()

    detector = load_face_detector()

    path_lower = args.path.lower()
    is_video = any(path_lower.endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"])

    if not is_video:
        image_bgr = cv2.imread(args.path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print("Failed to read image.", file=sys.stderr)
            sys.exit(2)
        feats_list, _ = analyze_faces_in_image(image_bgr, detector)
        scores = [compute_fake_score(f) for f in feats_list]
        overall = aggregate_scores(scores)
        print({"face_scores": [round(s, 3) for s in scores], "overall": round(overall, 3)})
    else:
        frames = sample_video_frames(args.path, max_frames=args.max_frames, step=args.step)
        per_frame_scores = []
        for frame in frames:
            feats_list, _ = analyze_faces_in_image(frame, detector)
            scores = [compute_fake_score(f) for f in feats_list]
            if scores:
                per_frame_scores.append(float(np.mean(scores)))
        overall = aggregate_scores(per_frame_scores)
        print({"frame_scores": [round(s, 3) for s in per_frame_scores], "overall": round(overall, 3)})


if __name__ == "__main__":
    # If launched by Streamlit, do not run CLI
    is_streamlit = any("streamlit" in (arg or "") for arg in sys.argv)
    if is_streamlit:
        run_streamlit_app()
    else:
        # Allow `python app.py <path>` usage
        if len(sys.argv) > 1:
            run_cli()
        else:
            print("Run the Streamlit app with: streamlit run app.py\n" \
                  "Or run CLI: python app.py <image_or_video_path>")

