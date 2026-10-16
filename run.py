"""VinFast DriverGuard: canh bao tai xe dung dien thoai va hut thuoc khi lai.

Chay:  python run.py            (nhan q de thoat)
Do khong cua so:  python run.py --no-show --frames 150
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np

from dms.dashboard import Dashboard
from dms.events import EventLog
from dms.perception import Perception
from dms.rules import NORMAL, UNSURE, classify
from dms.smoking import SmokingDetector
from dms.temporal import AlertFilter
from dms.voice import Voice

LOW_LIGHT = 40  # do sang trung binh < 40/255 -> khong tin ket qua


def draw_geometry(frame, dec):
    """Ve phan hinh hoc ma luat dang dung, de giam khao thay truc quan:
    duong dien thoai -> tai (so = khoang cach / do rong mat) va cho kep tay -> mieng."""
    e = dec.extra
    if "ear_pt" in e:
        a, b = tuple(e["ear_pt"].astype(int)), tuple(e["phone_c"].astype(int))
        cv2.line(frame, a, b, (0, 230, 255), 2)
        mid = ((a[0] + b[0]) // 2 + 6, (a[1] + b[1]) // 2)
        cv2.putText(frame, f"tai {dec.ear_dist:.2f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2)
        if dec.phone is not None:
            x1, y2 = int(dec.phone[0]), int(dec.phone[3])
            cv2.putText(frame, f"u {e['u']:+.2f}  v {e['v']:+.2f}  w {e['pw']:.2f}", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 2)
    if "grip" in e and e["mouth_d"] < 1.0:  # chi ve khi tay da gan mat
        a, b = tuple(np.asarray(e["grip"]).astype(int)), tuple(np.asarray(e["mouth"]).astype(int))
        cv2.line(frame, a, b, (230, 80, 200), 2)
        cv2.circle(frame, a, 6, (230, 80, 200), -1)
        cv2.putText(frame, f"mieng {e['mouth_d']:.2f}", (a[0] + 8, a[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (230, 80, 200), 2)


def draw_overlays(frame, per, dec, cigs, pens):
    """Ve len anh camera: box dien thoai/thuoc la, moc tay, moc tai va mieng."""
    for x1, y1, x2, y2, c, name in cigs + pens:
        col = (220, 80, 170) if name in ("cigarette", "cigar") else (200, 200, 200)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(frame, f"{name} {c:.2f}", (int(x1), int(y2) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    for p in per["phones"]:
        x1, y1, x2, y2, c = p
        chosen = dec.phone is not None and p == dec.phone
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (60, 60, 230) if chosen else (150, 150, 150), 2)
        cv2.putText(frame, f"phone {c:.2f}", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 230), 2)
    for hand in per["hands"]:
        for x, y in hand.astype(int):
            cv2.circle(frame, (x, y), 2, (0, 230, 120), -1)
    face = per["face"]
    if face is not None:
        col = (255, 200, 0) if per["face_src"] == "thay" else (180, 180, 180)
        for pt in (face.ear_a, face.ear_b):
            cv2.circle(frame, tuple(pt.astype(int)), 6, col, 2)  # vong tron = moc tai
        cv2.circle(frame, tuple(face.mouth.astype(int)), 4, col, -1)
    draw_geometry(frame, dec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--save-ui", default="", help="luu anh dashboard cuoi cung ra file (de kiem tra)")
    ap.add_argument("--web", type=int, nargs="?", const=8000, default=0,
                    help="phat dashboard len trinh duyet http://localhost:PORT (mac dinh 8000), khong mo cua so")
    ap.add_argument("--yoloe", action="store_true",
                    help="bat them YOLOE tim box thuoc la (cham, ~150-380 ms/lan tren CPU, it khi bat duoc)")
    args = ap.parse_args()
    if args.web:
        args.no_show = True

    import torch
    # 6 nhan CPU: neu bat YOLOE thi 4 cho vong lap chinh, 2 cho tien trinh thuoc la
    torch.set_num_threads(4 if args.yoloe else 6)
    smoker = SmokingDetector() if args.yoloe else None
    per_fn = Perception(imgsz=args.imgsz, conf=args.conf)
    voice, filt, events, ui = Voice(), AlertFilter(), EventLog(), Dashboard()
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Khong mo duoc webcam {args.cam}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hub = None
    if args.web:
        from dms.webserver import FrameHub, serve
        hub = FrameHub()
        serve(hub, args.web)
        print(f"Mo trinh duyet: http://localhost:{args.web}  (dung: Ctrl+C)", flush=True)

    lat, loop_t, n, view = deque(maxlen=100), deque(maxlen=30), 0, None
    win = "VinFast DriverGuard - nhan q de thoat"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_cap = time.perf_counter()  # do tre tinh tu luc co anh den luc co quyet dinh
            frame = cv2.flip(frame, 1)  # soi guong cho de nhin khi tu thu
            per = per_fn(frame)
            face = per["face"]
            cigs, pens = [], []
            if smoker:
                # Tien trinh YOLOE chi lay khung moi nhat; ket qua co the tre ~0.5 s so voi khung nay
                smoker.submit(frame.copy(), face.mouth if face else None, face.width if face else None)
                cigs, pens = smoker.latest()
            dec = classify(per["phones"], per["hands"], face, cigs)
            raw = dec.state
            if per["brightness"] < LOW_LIGHT:
                raw, dec.reason = UNSURE, f"thiếu sáng ({per['brightness']:.0f} < {LOW_LIGHT})"
            alert, best, ratio = filt.update(t_cap, raw)
            lat.append((time.perf_counter() - t_cap) * 1000)
            loop_t.append(t_cap)
            n += 1

            if alert != NORMAL:
                voice.say(alert)  # tu chan doc lap lai trong 8 s
            draw_overlays(frame, per, dec, cigs, pens)
            events.update(t_cap, alert, frame)

            if not args.no_show or args.save_ui or hub:
                x = np.array(lat)
                stats = dict(fps=(len(loop_t) - 1) / (loop_t[-1] - loop_t[0]) if len(loop_t) > 1 else 0.0,
                             p50=np.percentile(x, 50), p95=np.percentile(x, 95),
                             brightness=per["brightness"], face=per["face_src"])
                view = ui.render(frame, alert, raw, best, ratio, dec.reason, events, stats)
            if hub:
                hub.publish(view)
            if not args.no_show:
                cv2.imshow(win, view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.frames and n >= args.frames:
                break
    except KeyboardInterrupt:
        pass  # Ctrl+C khi chay che do --web
    _shutdown(args, cap, smoker, voice, events, view, lat, n)


def _shutdown(args, cap, smoker, voice, events, view, lat, n):
    events.close(time.perf_counter())
    cap.release()
    if smoker:
        smoker.close()
    voice.close()
    cv2.destroyAllWindows()
    if args.save_ui and view is not None:
        cv2.imwrite(args.save_ui, view)
    if n > 10:
        x = np.array(list(lat)[5:])
        yoloe = f" | YOLOE lan cuoi {smoker.infer_ms:.0f} ms" if smoker else ""
        print(f"{n} khung | do tre anh->quyet dinh p50 {np.percentile(x, 50):.1f} ms, p95 {np.percentile(x, 95):.1f} ms"
              f"{yoloe} | su kien: {events.csv_path.name}")


if __name__ == "__main__":
    main()
