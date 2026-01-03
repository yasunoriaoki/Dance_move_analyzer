import base64
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from collections import deque

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
import streamlit as st

import UpperBody_LiveOverlay as ubo


def _get_mp_pose():
    try:
        return mp.solutions.pose
    except AttributeError as exc:
        try:
            from mediapipe.python.solutions import pose as pose_mod
        except Exception as import_exc:
            raise AttributeError(
                "mediapipe is missing solutions.pose; verify the mediapipe install."
            ) from import_exc
        return pose_mod


mp_pose = _get_mp_pose()


def _get_ffmpeg_path():
    local = shutil.which("ffmpeg")
    if local:
        return local
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


APP_TITLE = "Dance Practice Overlay"
OUTPUT_DIR = ".streamlit_outputs"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_uploaded_file(uploaded_file, dest_dir):
    ensure_dir(dest_dir)
    file_hash = hashlib.sha1(uploaded_file.getbuffer()).hexdigest()[:12]
    name, ext = os.path.splitext(uploaded_file.name)
    safe_name = f"{name}_{file_hash}{ext}"
    out_path = os.path.join(dest_dir, safe_name)
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = total_frames / fps if total_frames > 0 else 0.0
    return {
        "fps": fps,
        "frames": total_frames,
        "width": w,
        "height": h,
        "duration": duration,
    }


def read_frame_at_time(path, time_sec):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_frame = max(0, int(round(time_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _get_point(landmarks, lm_id, w, h):
    lm = landmarks[lm_id]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _midpoint(landmarks, left_id, right_id, w, h):
    left = _get_point(landmarks, left_id, w, h)
    right = _get_point(landmarks, right_id, w, h)
    return (left + right) / 2.0


def draw_pose_overlay(
    image_bgr,
    landmarks,
    connections,
    show_points=True,
    show_lines=True,
    show_guides=False,
    show_wrist_guides=False,
    point_radius=6,
    line_thickness=3,
):
    h, w = image_bgr.shape[:2]

    def _idx(node):
        return node.value if hasattr(node, "value") else int(node)

    if show_lines:
        for a, b in connections:
            la = landmarks[_idx(a)]
            lb = landmarks[_idx(b)]
            if hasattr(la, "visibility") and (la.visibility < 0.5 or lb.visibility < 0.5):
                continue
            ax, ay = int(la.x * w), int(la.y * h)
            bx, by = int(lb.x * w), int(lb.y * h)
            cv2.line(image_bgr, (ax, ay), (bx, by), (0, 255, 0), line_thickness)

    if show_points:
        for lid, lm in enumerate(landmarks):
            if hasattr(lm, "visibility") and lm.visibility < 0.5:
                continue
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image_bgr, (x, y), point_radius, (0, 0, 255), -1)

    if show_guides:
        left_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        right_shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        left_hip = mp_pose.PoseLandmark.LEFT_HIP.value
        right_hip = mp_pose.PoseLandmark.RIGHT_HIP.value

        mid_shoulder = _midpoint(landmarks, left_shoulder, right_shoulder, w, h)
        mid_hip = _midpoint(landmarks, left_hip, right_hip, w, h)
        y_shoulder = int(round(mid_shoulder[1]))
        y_hip = int(round(mid_hip[1]))
        cv2.line(image_bgr, (0, y_shoulder), (w - 1, y_shoulder), (0, 255, 255), 2)
        cv2.line(image_bgr, (0, y_hip), (w - 1, y_hip), (0, 255, 255), 2)

    if show_wrist_guides:
        left_wrist = mp_pose.PoseLandmark.LEFT_WRIST.value
        right_wrist = mp_pose.PoseLandmark.RIGHT_WRIST.value
        lw = landmarks[left_wrist]
        rw = landmarks[right_wrist]
        if (
            (not hasattr(lw, "visibility") or lw.visibility >= 0.5)
            and (not hasattr(rw, "visibility") or rw.visibility >= 0.5)
        ):
            lwp = _get_point(landmarks, left_wrist, w, h)
            rwp = _get_point(landmarks, right_wrist, w, h)
            y_mid = int(round((lwp[1] + rwp[1]) / 2.0))
            cv2.line(image_bgr, (0, y_mid), (w - 1, y_mid), (255, 200, 0), 2)
            cv2.line(
                image_bgr,
                (int(round(lwp[0])), 0),
                (int(round(rwp[0])), h - 1),
                (255, 200, 0),
                1,
            )


def render_pose_segment(
    in_path,
    out_path,
    start_sec,
    end_sec,
    show_points=True,
    show_lines=True,
    show_guides=False,
    show_wrist_guides=False,
    trace_enabled=True,
    trace_length=60,
    overlay_alpha=0.5,
    progress_cb=None,
):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = int(math.floor(end_sec * fps)) - 1
    if total_frames > 0:
        end_frame = min(end_frame, total_frames - 1)
    if end_frame < start_frame:
        raise ValueError("End time must be greater than start time.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    pose_connections = mp_pose.POSE_CONNECTIONS
    shoulder_trace = deque(maxlen=trace_length)
    hip_trace = deque(maxlen=trace_length)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = start_frame
        processed = 0
        total = max(1, end_frame - start_frame + 1)
        while frame_idx <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                overlay = frame.copy()
                draw_pose_overlay(
                    overlay,
                    res.pose_landmarks.landmark,
                    pose_connections,
                    show_points=show_points,
                    show_lines=show_lines,
                    show_guides=show_guides,
                    show_wrist_guides=show_wrist_guides,
                )
                if trace_enabled:
                    left_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                    right_shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                    left_hip = mp_pose.PoseLandmark.LEFT_HIP.value
                    right_hip = mp_pose.PoseLandmark.RIGHT_HIP.value
                    mid_shoulder = _midpoint(res.pose_landmarks.landmark, left_shoulder, right_shoulder, w, h)
                    mid_hip = _midpoint(res.pose_landmarks.landmark, left_hip, right_hip, w, h)
                    shoulder_trace.append((int(round(mid_shoulder[0])), int(round(mid_shoulder[1]))))
                    hip_trace.append((int(round(mid_hip[0])), int(round(mid_hip[1]))))
                    if len(shoulder_trace) >= 2:
                        cv2.polylines(
                            overlay,
                            [np.array(shoulder_trace, dtype=np.int32)],
                            False,
                            (255, 0, 0),
                            2,
                        )
                    if len(hip_trace) >= 2:
                        cv2.polylines(
                            overlay,
                            [np.array(hip_trace, dtype=np.int32)],
                            False,
                            (255, 0, 0),
                            2,
                        )
                frame = cv2.addWeighted(overlay, overlay_alpha, frame, 1.0 - overlay_alpha, 0.0)

            out.write(frame)
            frame_idx += 1
            processed += 1
            if progress_cb and processed % 5 == 0:
                progress_cb(min(1.0, processed / total))

    cap.release()
    out.release()


def render_upperbody_segment(
    in_path,
    out_path,
    start_sec,
    end_sec,
    show_guide_lines=True,
    show_shoulder_guide=True,
    show_hip_guide=True,
    show_wrist_guide=True,
    show_side_schematic=False,
    trace_enabled=True,
    trace_length=60,
    overlay_alpha=0.5,
    progress_cb=None,
):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = int(math.floor(end_sec * fps)) - 1
    if total_frames > 0:
        end_frame = min(end_frame, total_frames - 1)
    if end_frame < start_frame:
        raise ValueError("End time must be greater than start time.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    shoulder_trace = deque(maxlen=trace_length)
    hip_trace = deque(maxlen=trace_length)

    saved = {
        "SHOW_GUIDE_LINES": ubo.SHOW_GUIDE_LINES,
        "SHOW_SHOULDER_GUIDE": ubo.SHOW_SHOULDER_GUIDE,
        "SHOW_HIP_GUIDE": ubo.SHOW_HIP_GUIDE,
        "SHOW_WRIST_GUIDE": ubo.SHOW_WRIST_GUIDE,
        "SHOW_SIDE_SCHEMATIC": ubo.SHOW_SIDE_SCHEMATIC,
        "TRACE_ENABLED": ubo.TRACE_ENABLED,
        "TRACE_LENGTH": ubo.TRACE_LENGTH,
        "OVERLAY_ALPHA": ubo.OVERLAY_ALPHA,
    }
    ubo.SHOW_GUIDE_LINES = show_guide_lines
    ubo.SHOW_SHOULDER_GUIDE = show_shoulder_guide
    ubo.SHOW_HIP_GUIDE = show_hip_guide
    ubo.SHOW_WRIST_GUIDE = show_wrist_guide
    ubo.SHOW_SIDE_SCHEMATIC = show_side_schematic
    ubo.TRACE_ENABLED = trace_enabled
    ubo.TRACE_LENGTH = trace_length
    ubo.OVERLAY_ALPHA = overlay_alpha

    try:
        with ubo.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            frame_idx = start_frame
            processed = 0
            total = max(1, end_frame - start_frame + 1)
            while frame_idx <= end_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                if res.pose_landmarks:
                    overlay = frame.copy()
                    ubo.draw_selected(
                        overlay,
                        res.pose_landmarks,
                        ubo.INCLUDE_PARTS,
                        ubo.INCLUDE_PARTS_EDGES,
                    )
                    if ubo.TRACE_ENABLED:
                        mid_shoulder = ubo.estimate_midShoulder(res.pose_landmarks, w, h)
                        mid_hip = ubo.estimate_midHip(res.pose_landmarks, w, h)
                        shoulder_trace.append((int(round(mid_shoulder[0])), int(round(mid_shoulder[1]))))
                        hip_trace.append((int(round(mid_hip[0])), int(round(mid_hip[1]))))
                        if len(shoulder_trace) >= 2:
                            cv2.polylines(
                                overlay,
                                [np.array(shoulder_trace, dtype=np.int32)],
                                False,
                                ubo.TRACE_SHOULDER_COLOR,
                                ubo.TRACE_THICKNESS,
                            )
                        if len(hip_trace) >= 2:
                            cv2.polylines(
                                overlay,
                                [np.array(hip_trace, dtype=np.int32)],
                                False,
                                ubo.TRACE_HIP_COLOR,
                                ubo.TRACE_THICKNESS,
                            )
                    frame = cv2.addWeighted(
                        overlay,
                        ubo.OVERLAY_ALPHA,
                        frame,
                        1.0 - ubo.OVERLAY_ALPHA,
                        0.0,
                    )

                out.write(frame)
                frame_idx += 1
                processed += 1
                if progress_cb and processed % 5 == 0:
                    progress_cb(min(1.0, processed / total))
    finally:
        for key, value in saved.items():
            setattr(ubo, key, value)

        cap.release()
        out.release()


def merge_audio_segment(video_path, source_path, start_sec, end_sec):
    ffmpeg_path = _get_ffmpeg_path()
    if not ffmpeg_path:
        return False
    tmp_out = video_path + ".with_audio.mp4"
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        video_path,
        "-ss",
        str(start_sec),
        "-to",
        str(end_sec),
        "-i",
        source_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        tmp_out,
    ]
    result = subprocess.run(cmd, check=False, capture_output=True)
    if result.returncode != 0:
        return False
    try:
        os.replace(tmp_out, video_path)
    except OSError:
        return False
    return True


def load_video_as_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("ascii")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    st.markdown(
        "Upload a dance video, pick the segment you want to practice, then render an overlayed loop with audio."
    )

    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "mkv", "avi"])
    input_path = None
    info = None
    start_sec = None
    end_sec = None

    if uploaded:
        ensure_dir(OUTPUT_DIR)
        input_path = save_uploaded_file(uploaded, OUTPUT_DIR)
        info = get_video_info(input_path)

        st.subheader("Select Practice Segment")
        if info["duration"] > 0:
            start_sec, end_sec = st.slider(
                "Practice segment (sec)",
                min_value=0.0,
                max_value=float(info["duration"]),
                value=(0.0, min(info["duration"], 5.0)),
                step=0.1,
            )
        else:
            start_sec = 0.0
            end_sec = 0.0

        col_c, col_d = st.columns(2)
        with col_c:
            start_frame = read_frame_at_time(input_path, start_sec)
            if start_frame is not None:
                st.image(start_frame, caption="Start frame preview", width="stretch")
        with col_d:
            end_frame = read_frame_at_time(input_path, max(start_sec, end_sec - 0.05))
            if end_frame is not None:
                st.image(end_frame, caption="End frame preview", width="stretch")
    else:
        st.info("Upload a video to render a practice segment.")

    st.subheader("Overlay Settings")
    overlay_mode = st.selectbox(
        "Overlay style",
        ["Simple skeleton", "UpperBody overlay (from UpperBody_LiveOverlay.py)"],
        index=1,
    )
    show_lines = True
    show_points = True
    show_guides = False
    show_wrist_guides = False
    show_guide_lines = True
    show_side_schematic = False
    show_shoulder_guide = True
    show_hip_guide = True
    show_wrist_guide = True

    col1, col2, col3 = st.columns(3)
    with col1:
        if overlay_mode == "Simple skeleton":
            show_lines = st.checkbox("Show skeleton lines", value=True)
            show_points = st.checkbox("Show joints points", value=True)
        else:
            show_guide_lines = st.checkbox("Show guide lines", value=True)
            show_side_schematic = st.checkbox("Show side schematic", value=False)
    with col2:
        if overlay_mode == "Simple skeleton":
            show_guides = st.checkbox("Show shoulder/hip guides", value=True)
            show_wrist_guides = st.checkbox("Show wrist guides", value=False)
        else:
            show_shoulder_guide = st.checkbox("Show shoulder guides", value=True)
            show_hip_guide = st.checkbox("Show hip guides", value=True)
            show_wrist_guide = st.checkbox("Show wrist guides", value=True)
    with col3:
        trace_enabled = st.checkbox("Show traces", value=True)
        trace_length = st.slider("Trace length", 10, 200, 60, step=5)

    overlay_alpha = st.slider("Overlay strength", 0.1, 1.0, 0.5, step=0.05)
    loop_playback = st.checkbox("Loop playback", value=True)

    if input_path and st.button("Render overlay segment"):
        if end_sec <= start_sec:
            st.error("End time must be greater than start time.")
            return

        if overlay_mode == "Simple skeleton":
            settings_key = (
                f"{os.path.basename(input_path)}_{start_sec:.2f}_{end_sec:.2f}"
                f"_{int(show_lines)}{int(show_points)}{int(show_guides)}{int(show_wrist_guides)}"
                f"_{int(trace_enabled)}_{trace_length}_{overlay_alpha:.2f}"
            )
        else:
            settings_key = (
                f"{os.path.basename(input_path)}_{start_sec:.2f}_{end_sec:.2f}"
                f"_{int(show_guide_lines)}{int(show_shoulder_guide)}{int(show_hip_guide)}"
                f"{int(show_wrist_guide)}{int(show_side_schematic)}"
                f"_{int(trace_enabled)}_{trace_length}_{overlay_alpha:.2f}"
            )
        safe_key = hashlib.sha1(settings_key.encode("ascii")).hexdigest()[:12]
        out_path = os.path.join(OUTPUT_DIR, f"overlay_{safe_key}.mp4")

        progress = st.progress(0.0)
        try:
            if overlay_mode == "Simple skeleton":
                render_pose_segment(
                    input_path,
                    out_path,
                    start_sec,
                    end_sec,
                    show_points=show_points,
                    show_lines=show_lines,
                    show_guides=show_guides,
                    show_wrist_guides=show_wrist_guides,
                    trace_enabled=trace_enabled,
                    trace_length=trace_length,
                    overlay_alpha=overlay_alpha,
                    progress_cb=lambda p: progress.progress(p),
                )
            else:
                render_upperbody_segment(
                    input_path,
                    out_path,
                    start_sec,
                    end_sec,
                    show_guide_lines=show_guide_lines,
                    show_shoulder_guide=show_shoulder_guide,
                    show_hip_guide=show_hip_guide,
                    show_wrist_guide=show_wrist_guide,
                    show_side_schematic=show_side_schematic,
                    trace_enabled=trace_enabled,
                    trace_length=trace_length,
                    overlay_alpha=overlay_alpha,
                    progress_cb=lambda p: progress.progress(p),
                )
        except Exception as exc:
            st.error(f"Render failed: {exc}")
            return
        finally:
            progress.progress(1.0)

        audio_ok = merge_audio_segment(out_path, input_path, start_sec, end_sec)
        if not audio_ok:
            st.warning("Audio merge failed or ffmpeg not found. Video will play without audio.")

        st.subheader("Overlay Preview")
        if loop_playback:
            b64 = load_video_as_base64(out_path)
            html = f"""
            <video controls loop autoplay muted="false" style="width:100%; max-width:960px;">
              <source src="data:video/mp4;base64,{b64}" type="video/mp4">
            </video>
            """
            st.components.v1.html(html, height=540)
        else:
            with open(out_path, "rb") as f:
                st.video(f.read())

        st.success(f"Saved overlay video: {out_path}")


if __name__ == "__main__":
    main()
