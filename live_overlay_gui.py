import tkinter as tk
from collections import deque
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import UpperBody_LiveOverlay as ubo


class LiveOverlayApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Upper Body Live Overlay")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.geometry("1000x800")
        self.root.resizable(True, True)

        self.camera_index = tk.IntVar(value=0)
        self.show_guide_lines = tk.BooleanVar(value=True)
        self.show_shoulder_guide = tk.BooleanVar(value=True)
        self.show_hip_guide = tk.BooleanVar(value=True)
        self.show_wrist_guide = tk.BooleanVar(value=True)
        self.show_side_schematic = tk.BooleanVar(value=False)
        self.trace_enabled = tk.BooleanVar(value=True)
        self.trace_length = tk.IntVar(value=ubo.TRACE_LENGTH)
        self.overlay_alpha = tk.DoubleVar(value=ubo.OVERLAY_ALPHA)

        self.running = False
        self.cap = None
        self.pose = None
        self.shoulder_trace = deque(maxlen=ubo.TRACE_LENGTH)
        self.hip_trace = deque(maxlen=ubo.TRACE_LENGTH)
        self.video_window = None
        self.video_label = None
        self.video_container = None
        self.current_image = None
        self.target_size = None
        self.video_row_index = None

        self._build_ui()

    def _build_ui(self):
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)

        row = 0
        tk.Label(frame, text="Camera index:").grid(row=row, column=0, sticky="w")
        tk.Entry(frame, textvariable=self.camera_index, width=6).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Checkbutton(frame, text="Show guide lines", variable=self.show_guide_lines).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        tk.Checkbutton(frame, text="Show shoulder guides", variable=self.show_shoulder_guide).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        tk.Checkbutton(frame, text="Show hip guides", variable=self.show_hip_guide).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        tk.Checkbutton(frame, text="Show wrist guides", variable=self.show_wrist_guide).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        tk.Checkbutton(frame, text="Show side schematic", variable=self.show_side_schematic).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        tk.Checkbutton(frame, text="Show traces", variable=self.trace_enabled).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1

        tk.Label(frame, text="Trace length").grid(row=row, column=0, sticky="w")
        tk.Scale(
            frame,
            variable=self.trace_length,
            from_=10,
            to=200,
            orient="horizontal",
            length=200,
        ).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(frame, text="Overlay strength").grid(row=row, column=0, sticky="w")
        tk.Scale(
            frame,
            variable=self.overlay_alpha,
            from_=0.1,
            to=1.0,
            resolution=0.05,
            orient="horizontal",
            length=200,
        ).grid(row=row, column=1, sticky="w")
        row += 1

        btn_frame = tk.Frame(frame, pady=8)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky="w")
        tk.Button(btn_frame, text="Start", command=self.start).pack(side="left", padx=(0, 8))
        tk.Button(btn_frame, text="Stop", command=self.stop).pack(side="left")

        row += 1
        self.video_row_index = row
        self.video_window = tk.Toplevel(self.root)
        self.video_window.title("Live Overlay Video")
        self.video_window.geometry("960x540")
        self.video_window.resizable(True, True)
        self.video_window.protocol("WM_DELETE_WINDOW", self._hide_video_window)

        self.video_container = tk.Frame(self.video_window, bg="black")
        self.video_container.pack(fill="both", expand=True)
        self.video_container.pack_propagate(False)
        self.video_label = tk.Label(self.video_container, bg="black")
        self.video_label.pack(fill="both", expand=True)
        self.video_container.bind("<Configure>", self._on_video_resize)

    def _on_video_resize(self, event):
        if event.width > 0 and event.height > 0:
            self.target_size = (event.width, event.height)

    def _hide_video_window(self):
        if self.video_window:
            self.video_window.withdraw()

        self.root.resizable(True, True)

    def _apply_settings(self, trace_len):
        ubo.SHOW_GUIDE_LINES = self.show_guide_lines.get()
        ubo.SHOW_SHOULDER_GUIDE = self.show_shoulder_guide.get()
        ubo.SHOW_HIP_GUIDE = self.show_hip_guide.get()
        ubo.SHOW_WRIST_GUIDE = self.show_wrist_guide.get()
        ubo.SHOW_SIDE_SCHEMATIC = self.show_side_schematic.get()
        ubo.TRACE_ENABLED = self.trace_enabled.get()
        ubo.TRACE_LENGTH = trace_len
        ubo.OVERLAY_ALPHA = float(self.overlay_alpha.get())

    def start(self):
        if self.running:
            return
        if self.video_window:
            self.video_window.deiconify()
        cam_index = int(self.camera_index.get())
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", f"Failed to open camera {cam_index}")
            return

        self.cap = cap
        self.pose = ubo.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.running = True
        self.root.after(1, self._update_frame)

    def stop(self):
        self.running = False
        if self.pose:
            self.pose.close()
            self.pose = None
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_label:
            self.video_label.configure(image="")
            self.current_image = None
        if self.video_window:
            self.video_window.withdraw()

    def on_close(self):
        self.stop()
        self.root.destroy()

    def _update_frame(self):
        if not self.running or not self.cap or not self.pose:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.stop()
            return

        trace_len = self.trace_length.get()
        if trace_len != ubo.TRACE_LENGTH:
            self.shoulder_trace = deque(maxlen=trace_len)
            self.hip_trace = deque(maxlen=trace_len)

        self._apply_settings(trace_len)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if res.pose_landmarks:
            overlay = frame.copy()
            ubo.draw_selected(
                overlay,
                res.pose_landmarks,
                ubo.INCLUDE_PARTS,
                ubo.INCLUDE_PARTS_EDGES,
            )
            if ubo.TRACE_ENABLED:
                h, w = frame.shape[:2]
                mid_shoulder = ubo.estimate_midShoulder(res.pose_landmarks, w, h)
                mid_hip = ubo.estimate_midHip(res.pose_landmarks, w, h)
                self.shoulder_trace.append((int(round(mid_shoulder[0])), int(round(mid_shoulder[1]))))
                self.hip_trace.append((int(round(mid_hip[0])), int(round(mid_hip[1]))))
                if len(self.shoulder_trace) >= 2:
                    cv2.polylines(
                        overlay,
                        [np.array(self.shoulder_trace, dtype=np.int32)],
                        False,
                        ubo.TRACE_SHOULDER_COLOR,
                        ubo.TRACE_THICKNESS,
                    )
                if len(self.hip_trace) >= 2:
                    cv2.polylines(
                        overlay,
                        [np.array(self.hip_trace, dtype=np.int32)],
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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.target_size:
            target_w, target_h = self.target_size
            if target_w > 0 and target_h > 0:
                h, w = frame_rgb.shape[:2]
                scale = min(target_w / float(w), target_h / float(h))
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                if new_w != w or new_h != h:
                    frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(frame_rgb)
        self.current_image = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=self.current_image)

        self.root.after(1, self._update_frame)


if __name__ == "__main__":
    app = LiveOverlayApp()
    app.root.mainloop()
