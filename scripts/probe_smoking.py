"""Do xem YOLOE nhin thay gi o vung mieng (khong loc lop, nguong rat thap).

Chay roi ngam/cam vat gia lam thuoc la truoc mieng trong ~10 giay:
    python scripts/probe_smoking.py
In ra cac lop YOLOE doan nhieu nhat va luu anh vung cat vao logs/probe/.
"""
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dms.perception import Perception  # noqa: E402
from dms.smoking import MODELS, _crop  # noqa: E402

DURATION_S = 10
OUT = Path(__file__).resolve().parent.parent / "logs" / "probe"


def main():
    from ultralytics import YOLOE
    OUT.mkdir(parents=True, exist_ok=True)
    yoloe = YOLOE(str(MODELS / "yoloe-11s-seg-pf.pt"))
    per_fn = Perception()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count, best = Counter(), defaultdict(float)
    t0, i = time.perf_counter(), 0
    print(f"Ngam/cam vat truoc mieng trong {DURATION_S} giay...")
    while time.perf_counter() - t0 < DURATION_S:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        face = per_fn(frame)["face"]
        _, _, img = _crop(frame, face.mouth if face else None, face.width if face else None)
        r = yoloe.predict(img, imgsz=320, conf=0.03, verbose=False)[0]
        for k, c in zip(r.boxes.cls.cpu().numpy().astype(int).tolist(), r.boxes.conf.cpu().numpy().tolist()):
            name = yoloe.names[k]
            count[name] += 1
            best[name] = max(best[name], c)
        cv2.imwrite(str(OUT / f"crop_{i:02d}.jpg"), r.plot())
        i += 1
    cap.release()
    print(f"{i} lan chay. Cac lop xuat hien nhieu nhat (so lan, conf cao nhat):")
    for name, n in count.most_common(12):
        print(f"  {name:20s} {n:3d}  max conf {best[name]:.2f}")
    print(f"Anh vung cat da luu o {OUT}")


if __name__ == "__main__":
    main()
