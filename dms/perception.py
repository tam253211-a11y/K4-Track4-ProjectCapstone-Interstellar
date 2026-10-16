"""Chay model tren mot khung hinh: YOLO (dien thoai) + MediaPipe (tay, mat)."""
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from ultralytics import YOLO

from .rules import Face

MODELS = Path(__file__).resolve().parent.parent / "models"
PHONE_CLASS = 67  # "cell phone" trong COCO
FACE_MEMORY_S = 1.5  # dien thoai che mat thi dung lai vi tri mat cu trong 1.5 s


class Perception:
    def __init__(self, imgsz=480, conf=0.25):
        self.imgsz, self.conf = imgsz, conf
        self.yolo = YOLO(str(MODELS / "yolo11n.pt"))
        # Nap model bang bytes: MediaPipe crash khi duong dan co dau tieng Viet ("Buổi").
        self.hand_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=(MODELS / "hand_landmarker.task").read_bytes()),
            running_mode=vision.RunningMode.VIDEO, num_hands=2))
        self.face_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=(MODELS / "face_landmarker.task").read_bytes()),
            running_mode=vision.RunningMode.VIDEO, num_faces=1))
        self.t0 = time.perf_counter()
        self.last_face, self.last_face_t = None, -1e9

    def __call__(self, frame):
        h, w = frame.shape[:2]
        now = time.perf_counter()
        ts = int((now - self.t0) * 1000)

        r = self.yolo.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=[PHONE_CLASS], verbose=False)[0]
        phones = [(*b, c) for b, c in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist())]

        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        scale = np.array([w, h], dtype=float)
        hands = [np.array([[p.x, p.y] for p in hl]) * scale for hl in self.hand_lm.detect_for_video(img, ts).hand_landmarks]
        faces = self.face_lm.detect_for_video(img, ts).face_landmarks

        face_pts, face_src = None, "khong"
        if faces:
            face_pts = np.array([[p.x, p.y] for p in faces[0]]) * scale
            self.last_face, self.last_face_t = face_pts, now
            face_src = "thay"
        elif self.last_face is not None and now - self.last_face_t < FACE_MEMORY_S:
            face_pts, face_src = self.last_face, "nho"

        face = Face.from_landmarks(face_pts) if face_pts is not None else None
        return dict(phones=phones, hands=hands, face=face, face_pts=face_pts, face_src=face_src,
                    latency_ms=(time.perf_counter() - now) * 1000)
