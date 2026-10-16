"""Buoc 1: kiem tra webcam + YOLO (dien thoai) + MediaPipe (tay, mat) chay duoc tren may nay.

Chay co cua so:   python scripts/step1_check.py
Chay khong cua so, do FPS:   python scripts/step1_check.py --no-show --frames 150
Nhan q de thoat.
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
PHONE_CLASS = 67  # "cell phone" trong COCO


def build_landmarkers():
    # Nap model bang bytes: MediaPipe crash khi duong dan co dau tieng Viet ("Buổi").
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=(MODELS / "hand_landmarker.task").read_bytes()),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=(MODELS / "face_landmarker.task").read_bytes()),
        running_mode=vision.RunningMode.VIDEO, num_faces=1))
    return hand, face


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--frames", type=int, default=0, help="0 = chay den khi nhan q")
    args = ap.parse_args()

    yolo = YOLO(str(MODELS / "yolo11n.pt") if (MODELS / "yolo11n.pt").exists() else "yolo11n.pt")
    hand_lm, face_lm = build_landmarkers()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Khong mo duoc webcam {args.cam}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    t_yolo, t_hand, t_face, t_total = [], [], [], []
    n, phone_frames = 0, 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int((time.perf_counter() - t0) * 1000)
        a = time.perf_counter()

        r = yolo.predict(frame, imgsz=args.imgsz, conf=args.conf, classes=[PHONE_CLASS], verbose=False)[0]
        b = time.perf_counter()
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hands = hand_lm.detect_for_video(img, ts)
        c = time.perf_counter()
        faces = face_lm.detect_for_video(img, ts)
        d = time.perf_counter()

        t_yolo.append(b - a); t_hand.append(c - b); t_face.append(d - c); t_total.append(d - a)
        n += 1
        phone_frames += len(r.boxes) > 0

        if not args.no_show:
            h, w = frame.shape[:2]
            for box, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"phone {cf:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            for hl in hands.hand_landmarks:
                for p in hl:
                    cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, (0, 255, 0), -1)
            for fl in faces.face_landmarks:
                for p in fl[::4]:
                    cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (255, 200, 0), -1)
            fps = 1.0 / np.mean(t_total[-30:])
            cv2.putText(frame, f"FPS {fps:.1f} | tay {len(hands.hand_landmarks)} | mat {len(faces.face_landmarks)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Buoc 1 - nhan q de thoat", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.frames and n >= args.frames:
            break

    cap.release()
    cv2.destroyAllWindows()
    if n < 10:
        raise SystemExit("Qua it khung hinh de do")

    def ms(xs):
        xs = np.array(xs[5:]) * 1000  # bo vai khung dau (khoi dong)
        return f"p50 {np.percentile(xs, 50):6.1f} ms | p95 {np.percentile(xs, 95):6.1f} ms"

    print(f"Khung hinh: {n}, khung co dien thoai: {phone_frames}")
    print(f"YOLO      : {ms(t_yolo)}")
    print(f"Hand      : {ms(t_hand)}")
    print(f"Face      : {ms(t_face)}")
    print(f"Tong      : {ms(t_total)}  -> ~{1000 / np.percentile(np.array(t_total[5:]) * 1000, 50):.1f} FPS")


if __name__ == "__main__":
    main()
