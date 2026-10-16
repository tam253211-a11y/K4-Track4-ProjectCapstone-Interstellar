"""VinFast DriverGuard: canh bao tai xe dung dien thoai khi lai.

Chay:  python run.py            (nhan q de thoat)
Do khong cua so:  python run.py --no-show --frames 150
"""
import argparse

import cv2
import numpy as np

from dms.perception import Perception
from dms.rules import CALLING, NORMAL, TEXTING, UNSURE, classify

COLORS = {NORMAL: (0, 170, 0), CALLING: (0, 0, 255), TEXTING: (0, 120, 255), UNSURE: (0, 200, 255)}


def draw(frame, per, dec, fps):
    for p in per["phones"]:
        x1, y1, x2, y2, c = p
        chosen = dec.phone is not None and p == dec.phone
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255) if chosen else (150, 150, 150), 2)
        cv2.putText(frame, f"phone {c:.2f}", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    for hand in per["hands"]:
        for x, y in hand.astype(int):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    face = per["face"]
    if face is not None:
        col = (255, 200, 0) if per["face_src"] == "thay" else (180, 180, 180)
        for pt in (face.ear_a, face.ear_b):
            cv2.circle(frame, tuple(pt.astype(int)), 6, col, 2)  # vong tron = moc tai
        cv2.circle(frame, tuple(face.mouth.astype(int)), 4, col, -1)

    color = COLORS[dec.state]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), color, -1)
    cv2.putText(frame, dec.state, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    ear = "-" if dec.ear_dist is None else f"{dec.ear_dist:.2f}"
    cv2.putText(frame, f"{dec.reason} | tai {ear} | mat: {per['face_src']} | {fps:.1f} FPS",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    args = ap.parse_args()

    per_fn = Perception(imgsz=args.imgsz, conf=args.conf)
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Khong mo duoc webcam {args.cam}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    lat, n = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # soi guong cho de nhin khi tu thu
        per = per_fn(frame)
        dec = classify(per["phones"], per["hands"], per["face"])
        lat.append(per["latency_ms"])
        n += 1
        if not args.no_show:
            draw(frame, per, dec, 1000 / np.mean(lat[-30:]))
            cv2.imshow("VinFast DriverGuard - nhan q de thoat", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.frames and n >= args.frames:
            break

    cap.release()
    cv2.destroyAllWindows()
    if n > 10:
        x = np.array(lat[5:])
        print(f"{n} khung | latency p50 {np.percentile(x, 50):.1f} ms, p95 {np.percentile(x, 95):.1f} ms")


if __name__ == "__main__":
    main()
