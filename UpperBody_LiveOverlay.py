import cv2
import mediapipe as mp
import numpy as np
import math
import os
import subprocess
from collections import deque
import argparse


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



def get_point(pl, lm, w, h):
    p = pl.landmark[lm]
    return np.array([p.x * w, p.y * h], dtype=np.float32)

def estimate_chin_top(pose_landmarks, w, h):
    innerEye_L = get_point(pose_landmarks, mp_pose.PoseLandmark.LEFT_EYE_INNER, w, h)
    innerEye_R = get_point(pose_landmarks, mp_pose.PoseLandmark.RIGHT_EYE_INNER, w, h)
    innerEye_M=(innerEye_L+innerEye_R)/2
    
    ml = get_point(pose_landmarks, mp_pose.PoseLandmark.MOUTH_LEFT, w, h)
    mr = get_point(pose_landmarks, mp_pose.PoseLandmark.MOUTH_RIGHT, w, h)
    mouth_mid = (ml + mr) / 2.0

    # direction from nose to mouth
    v = mouth_mid - innerEye_M
    chin = mouth_mid + 0.8 * v   # tune 0.6~1.2 depending on camera / person

    return chin

def estimate_head_top(pose_landmarks, w, h):
    leye = get_point(pose_landmarks, mp_pose.PoseLandmark.LEFT_EYE, w, h)
    reye = get_point(pose_landmarks, mp_pose.PoseLandmark.RIGHT_EYE, w, h)
    eye_mid = (leye + reye) / 2.0

    #nose = get_point(pose_landmarks, mp_pose.PoseLandmark.NOSE, w, h)
    innerEye_L = get_point(pose_landmarks, mp_pose.PoseLandmark.LEFT_EYE_INNER, w, h)
    innerEye_R = get_point(pose_landmarks, mp_pose.PoseLandmark.RIGHT_EYE_INNER, w, h)
    innerEye_M=(innerEye_L+innerEye_R)/2

    ml = get_point(pose_landmarks, mp_pose.PoseLandmark.MOUTH_LEFT, w, h)
    mr = get_point(pose_landmarks, mp_pose.PoseLandmark.MOUTH_RIGHT, w, h)
    mouth_mid = (ml + mr) / 2.0
    
    # "up" direction: from nose toward eyes, then extend beyond eyes
    up_dir = eye_mid - mouth_mid
    norm = np.linalg.norm(up_dir)
    if norm < 1e-6:
        return None
    up_dir = up_dir / norm

    # scale using inter-eye distance
    scale = np.linalg.norm(leye - reye)
    head_top = eye_mid + 2.0 * scale * up_dir   # tune 1.5~2.8
    return head_top

def estimate_solar_plexus(pose_landmarks, w, h):
    lh = get_point(pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
    rh = get_point(pose_landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
    mid_hip = (lh + rh) / 2.0

    ls = get_point(pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
    rs = get_point(pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
    mid_shoulder = (ls + rs) / 2.0
    
    # point between shoulders and hips
    solar_plexus = mid_shoulder + 0.25 * (mid_hip - mid_shoulder)  # tweak 0.25~0.45

    return(solar_plexus)


def estimate_midHip(pose_landmarks, w, h):
    
    RH=get_point(pose_landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)

    LH=get_point(pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)

    mid_hip=(LH + RH) / 2.0

    return(mid_hip)

def estimate_midShoulder(pose_landmarks, w, h):
    
    RS=get_point(pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)

    LS=get_point(pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)

    midShoulder=(LS + RS) / 2.0

    return(midShoulder)
    
def estimate_mizoochi(pose_landmarks, w, h):
    RS=get_point(pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
    x_RS, y_RS = int(RS[0]), int(RS[1])
    
    LS=get_point(pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
    x_LS, y_LS = int(LS[0]), int(LS[1])

    LH=get_point(pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
    x_LH, y_LH = int(LH[0]), int(LH[1])

    x_delta=x_RS-x_LS
    y_delta=y_RS-y_LS

    length=(y_LS-y_LH)/4

    
    if(x_delta==0):
        ForB=1
        x_delta=1
        length=1
    else:
        ForB=abs(x_delta)/(x_delta)
    
    x_midS=(x_RS+x_LS)/2
    y_midS=(y_RS+y_LS)/2
    
    weight=1
    
    x_mizoochi=(x_midS+ForB*(y_delta)*length/abs(x_delta))
    y_mizoochi=(y_midS-ForB*(x_delta)*length/abs(x_delta))
    
    return([x_mizoochi,y_mizoochi])
     


def estimate_heso(pose_landmarks, w, h):
    RH=get_point(pose_landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
    x_RH, y_RH = int(RH[0]), int(RH[1])
    
    LH=get_point(pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
    x_LH, y_LH = int(LH[0]), int(LH[1])

    LS=get_point(pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
    x_LS, y_LS = int(LS[0]), int(LS[1])

    x_delta=x_RH-x_LH
    y_delta=y_RH-y_LH

    length=(y_LS-y_LH)/4

    if(x_delta==0):
        ForB=1
        x_delta=1
        length=1
    else:
        ForB=abs(x_delta)/(x_delta)
    
    x_midH=(x_RH+x_LH)/2
    y_midH=(y_RH+y_LH)/2
    
    
    x_heso=(x_midH-ForB*(y_delta)*length/abs(x_delta))
    y_heso=(y_midH+ForB*(x_delta)*length/abs(x_delta))
    
    return([x_heso,y_heso])
     

def linFuncFromPoints(point_1, point_2):
    x_11=point_1[0]
    x_12=point_2[0]
    y_11=point_1[1]
    y_12=point_2[1]

    if(x_11-x_12==0):
        a1=0
        b1=0

    else:
        b1=(y_12*x_11-y_11*x_12)/(x_11-x_12)
        a1=(y_11-y_12)/(x_11-x_12)

    return([a1,b1])

def crossPointFromLinFuncS(a1,b1, a2,b2):

    if(a1==b1):
        x=0
        y=0
    else:
        x=(b2-b1)/(a1-a2)
        y=a1*x+b1

    return([x,y])

def crossPointFromPoints(point_11, point_12,point_21, point_22):
    a1, b1 = linFuncFromPoints(point_11, point_12)
    a2, b2 = linFuncFromPoints(point_21, point_22)

    if((a1==b1 and b1==0) and (a2==b2 and b2==0)):
        crossPoints=[0,0]
    elif ((a1==b1 and b1==0)):
        a,b = linFuncFromPoints(point_21, point_22)
        crossPoints[point_11[0],a*point_11[0]+b]

    elif ((a2==b2 and b2==0)):
        a,b = linFuncFromPoints(point_11, point_12)
        crossPoints[point_21[0],a*point_21[0]+b]
    else:
        crossPoints=crossPointFromLinFuncS(a1,b1, a2,b2)

    return(crossPoints)

def weightCenterTriangle(point_a,point_b,point_c):

    return(crossPointFromPoints(point_a, (point_b+point_c)/2,point_c,(point_a+point_b)/2))


def weightCenterLectangle(point_a,point_b,point_c,point_d): # be careful as it is sensitive to the order of abcd, need to be either clockwise or counter clockwise
    CT11=weightCenterTriangle(point_a,point_b,point_c)
    CT12=weightCenterTriangle(point_c,point_d,point_a)

    CT21=weightCenterTriangle(point_b,point_c,point_d)
    CT22=weightCenterTriangle(point_d,point_a,point_b)

    return(crossPointFromPoints(CT11,CT12,CT21,CT22))




mp_pose = _get_mp_pose()

INCLUDE_PARTS = [
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL,
]
INCLUDE_PARTS_EDGES = [
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL),
]
SHOW_GUIDE_LINES = True
SHOW_SHOULDER_GUIDE = True
SHOW_HIP_GUIDE = True
SHOW_WRIST_GUIDE = True
GUIDE_LINE_THICKNESS = 2
SHOW_SIDE_SCHEMATIC = False
SIDE_SCHEMATIC_SIZE = (160, 220)  # (width, height)
SIDE_SCHEMATIC_MARGIN = 12
SIDE_Z_SPAN = 10.0
TRACE_LENGTH = 60
TRACE_THICKNESS = 2
TRACE_SHOULDER_COLOR = (255, 0, 0)
TRACE_HIP_COLOR = (255, 0, 0)
TRACE_ENABLED = True
OVERLAY_ALPHA = 0.5

def _depth_color(z, z_min, z_range):
    norm = (z - z_min) / z_range
    norm = max(0.0, min(1.0, norm))
    r = int(255 * (1.0 - norm))
    b = int(255 * norm)
    return (b, 0, r)

def _nose_ear_scale_norm(landmarks, min_scale=1e-4):
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

    def dist(a, b):
        dz = a.z - b.z
        return float(np.sqrt(dz * dz))

    d_left = dist(nose, left_ear)
    d_right = dist(nose, right_ear)
    d_avg = 0.5 * (d_left + d_right)
    if d_avg < min_scale:
        return None
    return d_avg

def draw_selected(
    image_bgr,
    pose_landmarks,
    points,
    edges,
    point_radius=12,
    point_thickness=-1,
    line_thickness=6,
):
    h, w = image_bgr.shape[:2]
    thin_thickness = GUIDE_LINE_THICKNESS

    def draw_horizontal(y, color):
        yy = int(round(y))
        cv2.line(image_bgr, (0, yy), (w - 1, yy), color, thin_thickness)

    def draw_extended(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-6:
            xx = int(round(x1))
            cv2.line(
                image_bgr,
                (xx, 0),
                (xx, h - 1),
                (0, 0, 255),
                thin_thickness,
            )
            return
        m = dy / dx
        b = y1 - m * x1
        y_at_0 = b
        y_at_w = m * (w - 1) + b
        cv2.line(
            image_bgr,
            (0, int(round(y_at_0))),
            (w - 1, int(round(y_at_w))),
            (0, 0, 255),
            thin_thickness,
        )

    def draw_side_schematic():
        if not SHOW_SIDE_SCHEMATIC:
            return

        sw, sh = SIDE_SCHEMATIC_SIZE
        x0 = SIDE_SCHEMATIC_MARGIN
        y0 = h - sh - SIDE_SCHEMATIC_MARGIN
        x1 = x0 + sw
        y1 = y0 + sh
        if y0 < 0 or x1 > w:
            return

        roi = image_bgr[y0:y1, x0:x1]
        roi[:] = (245, 245, 245)
        cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (60, 60, 60), 1)

        ids = [
            mp_pose.PoseLandmark.NOSE,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_HEEL,
        ]

        points = {}
        for lid in ids:
            lm = pose_landmarks.landmark[lid.value]
            if hasattr(lm, "visibility") and lm.visibility < 0.5:
                continue
            points[lid] = lm

        if mp_pose.PoseLandmark.NOSE not in points or len(points) < 3:
            return

        z_scale = _nose_ear_scale_norm(pose_landmarks.landmark)
        if z_scale is None:
            z_scale = 1.0

        def z_scaled(lm):
            return lm.z / z_scale

        nose = points[mp_pose.PoseLandmark.NOSE]
        z_center = z_scaled(nose)
        z_range = SIDE_Z_SPAN
        pad = 8
        scale_w = sw - 2 * pad
        scale_h = sh - 2 * pad

        def project(lm):
            dx = (z_scaled(lm) - z_center) * scale_w / z_range
            sx = x0 + pad + scale_w * 0.5 + dx
            sy = y0 + pad + lm.y * scale_h
            return int(round(sx)), int(round(sy))

        def draw_seg(a, b, color, thickness):
            if a in points and b in points:
                ax, ay = project(points[a])
                bx, by = project(points[b])
                cv2.line(image_bgr, (ax, ay), (bx, by), color, thickness)

        z_min = z_center - z_range * 0.5
        line_color = (0, 120, 255)
        point_color = (30, 30, 30)
        line_thickness = 2

        segments = [
            (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        ]

        for a, b in segments:
            if a in points and b in points:
                avg_z = (z_scaled(points[a]) + z_scaled(points[b])) * 0.5
                color = _depth_color(avg_z, z_min, z_range)
                draw_seg(a, b, color, line_thickness)

        for lid, lm in points.items():
            px, py = project(lm)
            color = _depth_color(z_scaled(lm), z_min, z_range)
            cv2.circle(image_bgr, (px, py), 3, color, -1)

    # Draw edges (lines)
    for a, b in edges:
        la = pose_landmarks.landmark[a.value]
        lb = pose_landmarks.landmark[b.value]

        # Optional: skip if low visibility
        if hasattr(la, "visibility") and (la.visibility < 0.5 or lb.visibility < 0.5):
            continue

        ax, ay = int(la.x * w), int(la.y * h)
        bx, by = int(lb.x * w), int(lb.y * h)
        cv2.line(image_bgr, (ax, ay), (bx, by), (0, 255, 0), line_thickness)

    # Draw points (circles)
    for p in points:
        lm = pose_landmarks.landmark[p.value]
        if hasattr(lm, "visibility") and lm.visibility < 0.5:
            continue
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image_bgr, (x, y), point_radius, (0, 0, 255), point_thickness)

    
#    x_mizoochi=((x_largeSQ_1+x_largeSQ_2+x_largeSQ_3+x_largeSQ_4)/(4)+x_midS*weight)/(1+weight)
#    y_mizoochi=((y_largeSQ_1+y_largeSQ_2+y_largeSQ_3+y_largeSQ_4)/(4)+y_midS*weight)/(1+weight)

    midHip=estimate_midHip(pose_landmarks, w, h)

    cv2.circle(image_bgr, (round(midHip[0]),round(midHip[1])), point_radius, (0, 255, 0), point_thickness)
    if SHOW_GUIDE_LINES and SHOW_HIP_GUIDE:
        draw_horizontal(midHip[1], (0, 255, 0))

    mizoochi=estimate_mizoochi(pose_landmarks, w, h)
    cv2.circle(image_bgr, (round(mizoochi[0]),round(mizoochi[1])), point_radius, (0, 255, 0), point_thickness)

    heso=estimate_heso(pose_landmarks, w, h)
    cv2.circle(image_bgr, (round(heso[0]),round(heso[1])), point_radius, (0, 255, 0), point_thickness)

    cv2.line(image_bgr, (round(heso[0]),round(heso[1])), (round(mizoochi[0]),round(mizoochi[1])), (0, 255, 0), line_thickness)

    
    cv2.line(image_bgr, (round(midHip[0]),round(midHip[1])), (round(heso[0]),round(heso[1])), (0, 255, 0), line_thickness)


    head_top = estimate_head_top(pose_landmarks, w, h)

    cv2.circle(image_bgr, (round(head_top[0]),round(head_top[1])), point_radius, (0, 255,0), point_thickness)

    chin_top = estimate_chin_top(pose_landmarks, w, h)
    
    cv2.circle(image_bgr, (round(chin_top[0]),round(chin_top[1])), point_radius, (0, 255,0), point_thickness)

    midShoulder=estimate_midShoulder(pose_landmarks, w, h)

    cv2.circle(image_bgr, (round(midShoulder[0]),round(midShoulder[1])), point_radius, (0, 255,0), point_thickness)
    if SHOW_GUIDE_LINES and SHOW_SHOULDER_GUIDE:
        draw_horizontal(midShoulder[1], (0, 255, 0))


    cv2.line(image_bgr, (round(head_top[0]),round(head_top[1])), (round(chin_top[0]),round(chin_top[1])), (0, 255, 0), line_thickness)

    cv2.line(image_bgr, (round(mizoochi[0]),round(mizoochi[1])), (round(midShoulder[0]),round(midShoulder[1])), (0, 255, 0), line_thickness)

    cv2.line(image_bgr, (round(midShoulder[0]),round(midShoulder[1])), (round(chin_top[0]),round(chin_top[1])), (0, 255, 0), line_thickness)


    RH=get_point(pose_landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)

    LH=get_point(pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
    
    LS=get_point(pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
    
    RS=get_point(pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)

    if SHOW_GUIDE_LINES and SHOW_WRIST_GUIDE:
        rw = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        lw = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
        if (
            (not hasattr(rw, "visibility") or rw.visibility >= 0.5)
            and (not hasattr(lw, "visibility") or lw.visibility >= 0.5)
        ):
            rwp = get_point(pose_landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, w, h)
            lwp = get_point(pose_landmarks, mp_pose.PoseLandmark.LEFT_WRIST, w, h)
            mid_wrist = (rwp + lwp) / 2.0
            draw_extended(lwp, rwp)
            draw_horizontal(mid_wrist[1], (0, 255, 0))

    BC=bodyWeightCenter=weightCenterLectangle(LS,RS, RH, LH)

    cv2.circle(image_bgr, (round(BC[0]),round(BC[1])), point_radius, (255, 0,0), point_thickness)
    if SHOW_GUIDE_LINES and SHOW_HIP_GUIDE:
        draw_extended(LH, RH)
    if SHOW_GUIDE_LINES and SHOW_SHOULDER_GUIDE:
        draw_extended(LS, RS)
    draw_side_schematic()


def render_pose_on_video(in_path, out_path, max_frames=None):
    cap = cv2.VideoCapture(in_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    shoulder_trace = deque(maxlen=TRACE_LENGTH)
    hip_trace = deque(maxlen=TRACE_LENGTH)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if max_frames and n > max_frames:
                break
    
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
    
            if res.pose_landmarks:
                overlay = frame.copy()
                draw_selected(overlay, res.pose_landmarks, INCLUDE_PARTS, INCLUDE_PARTS_EDGES)
                if TRACE_ENABLED:
                    mid_shoulder = estimate_midShoulder(res.pose_landmarks, w, h)
                    mid_hip = estimate_midHip(res.pose_landmarks, w, h)
                    shoulder_trace.append((int(round(mid_shoulder[0])), int(round(mid_shoulder[1]))))
                    hip_trace.append((int(round(mid_hip[0])), int(round(mid_hip[1]))))
                    if len(shoulder_trace) >= 2:
                        cv2.polylines(
                            overlay,
                            [np.array(shoulder_trace, dtype=np.int32)],
                            False,
                            TRACE_SHOULDER_COLOR,
                            TRACE_THICKNESS,
                        )
                    if len(hip_trace) >= 2:
                        cv2.polylines(
                            overlay,
                            [np.array(hip_trace, dtype=np.int32)],
                            False,
                            TRACE_HIP_COLOR,
                            TRACE_THICKNESS,
                        )
                frame = cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1.0 - OVERLAY_ALPHA, 0.0)
    
            out.write(frame)
    
    cap.release()
    out.release()
    print("Wrote:", out_path)

    temp_out = out_path + ".with_audio.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            out_path,
            "-i",
            in_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            temp_out,
        ],
        check=False,
    )
    try:
        if os.path.exists(temp_out):
            os.replace(temp_out, out_path)
    except OSError:
        pass


def render_pose_on_webcam(camera_index=0):
    global SHOW_SHOULDER_GUIDE, SHOW_HIP_GUIDE, SHOW_WRIST_GUIDE, TRACE_ENABLED
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Failed to open webcam:", camera_index)
        return
    shoulder_trace = deque(maxlen=TRACE_LENGTH)
    hip_trace = deque(maxlen=TRACE_LENGTH)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                overlay = frame.copy()
                draw_selected(overlay, res.pose_landmarks, INCLUDE_PARTS, INCLUDE_PARTS_EDGES)
                if TRACE_ENABLED:
                    mid_shoulder = estimate_midShoulder(res.pose_landmarks, frame.shape[1], frame.shape[0])
                    mid_hip = estimate_midHip(res.pose_landmarks, frame.shape[1], frame.shape[0])
                    shoulder_trace.append((int(round(mid_shoulder[0])), int(round(mid_shoulder[1]))))
                    hip_trace.append((int(round(mid_hip[0])), int(round(mid_hip[1]))))
                    if len(shoulder_trace) >= 2:
                        cv2.polylines(
                            overlay,
                            [np.array(shoulder_trace, dtype=np.int32)],
                            False,
                            TRACE_SHOULDER_COLOR,
                            TRACE_THICKNESS,
                        )
                    if len(hip_trace) >= 2:
                        cv2.polylines(
                            overlay,
                            [np.array(hip_trace, dtype=np.int32)],
                            False,
                            TRACE_HIP_COLOR,
                            TRACE_THICKNESS,
                        )
                frame = cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1.0 - OVERLAY_ALPHA, 0.0)

            cv2.imshow("Upper Body Live Overlay", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                SHOW_SHOULDER_GUIDE = not SHOW_SHOULDER_GUIDE
            elif key == ord("h"):
                SHOW_HIP_GUIDE = not SHOW_HIP_GUIDE
            elif key == ord("w"):
                SHOW_WRIST_GUIDE = not SHOW_WRIST_GUIDE
            elif key == ord("t"):
                TRACE_ENABLED = not TRACE_ENABLED

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upper body overlay with optional video input.")
    parser.add_argument("--video", help="Path to input video file.")
    parser.add_argument("--output", help="Path to output video file (required if --video is set).")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument("--no-trace", action="store_true", help="Disable shoulder/hip trace lines.")
    parser.add_argument("--guides", action="store_true", help="Enable guide lines.")
    parser.add_argument("--no-guides", action="store_true", help="Disable guide lines.")
    parser.add_argument("--no-shoulder-guide", action="store_true", help="Disable shoulder guide lines.")
    parser.add_argument("--no-hip-guide", action="store_true", help="Disable hip guide lines.")
    parser.add_argument("--no-wrist-guide", action="store_true", help="Disable wrist guide lines.")
    args = parser.parse_args()

    if args.no_trace:
        TRACE_ENABLED = False
    if args.guides:
        SHOW_GUIDE_LINES = True
    if args.no_guides:
        SHOW_GUIDE_LINES = False
    if args.no_shoulder_guide:
        SHOW_SHOULDER_GUIDE = False
    if args.no_hip_guide:
        SHOW_HIP_GUIDE = False
    if args.no_wrist_guide:
        SHOW_WRIST_GUIDE = False

    if args.video:
        if not args.output:
            raise SystemExit("Missing --output for video rendering.")
        render_pose_on_video(args.video, args.output)
    else:
        render_pose_on_webcam(args.camera)
