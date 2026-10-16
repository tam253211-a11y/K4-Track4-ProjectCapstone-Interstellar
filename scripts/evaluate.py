"""Do dac co huong dan: lam lan luot tung tinh huong truoc webcam, script tu cham.

Chay:  python scripts/evaluate.py            (mac dinh 2 vong, ~4 phut)
       python scripts/evaluate.py --rounds 3
Ket qua: evidence/results.md va evidence/frames_<gio>.csv

Moi tinh huong: 3 s chuan bi (khong cham) + 8 s thuc hien (co cham).
- Do chinh xac tung khung: trang thai tho cua khung co dung voi tinh huong khong.
- Canh bao: tinh huong vi pham co bat dung canh bao khong, mat bao lau; tinh huong
  an toan co bi bao nham khong.
"""
import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dms.dashboard import FB, F  # noqa: E402
from dms.perception import Perception  # noqa: E402
from dms.rules import CALLING, NORMAL, SMOKING, TEXTING, UNSURE, classify  # noqa: E402
from dms.temporal import AlertFilter  # noqa: E402

PREP_S, ACT_S = 3.0, 8.0
LOW_LIGHT = 40
# (ma, huong dan, trang thai mong doi cua khung, canh bao mong doi)
SCENARIOS = [
    ("binh_thuong", "Ngồi bình thường, nhìn thẳng camera", NORMAL, NORMAL),
    ("goi_dien", "Áp điện thoại lên tai như đang gọi", CALLING, CALLING),
    ("nhan_tin", "Cầm điện thoại trước ngực, nhìn vào màn hình", TEXTING, TEXTING),
    ("liec_nhanh", "Nhấc điện thoại lên ~1 giây rồi bỏ xuống, lặp lại 3 lần", None, NORMAL),
    ("hut_thuoc", "Kẹp bút giữa ngón trỏ và ngón giữa, đưa lên miệng", SMOKING, SMOKING),
    ("che_mieng", "Xoè cả bàn tay che miệng như đang ngáp", NORMAL, NORMAL),
]


def overlay(frame, lines, color=(255, 255, 255)):
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.rectangle((0, 0, frame.shape[1], 30 + 28 * len(lines)), fill=(14, 20, 30))
    for i, (s, big) in enumerate(lines):
        d.text((12, 8 + 28 * i), s, font=FB[22] if big else F[16], fill=color if big else (200, 210, 225))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    out_dir = ROOT / "evidence"
    out_dir.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    per_fn = Perception()
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    win = "Do dac DriverGuard - nhan q de huy"
    # Lam nong model: lan chay dau rat cham, neu khong se an mat thoi gian cua tinh huong dau tien
    for _ in range(5):
        ok, frame = cap.read()
        if ok:
            per_fn(cv2.flip(frame, 1))

    rows, trials, lat = [], [], []
    plan = [(r + 1, s) for r in range(args.rounds) for s in SCENARIOS]
    aborted = False
    for idx, (rnd, (code, text, exp_raw, exp_alert)) in enumerate(plan):
        filt = AlertFilter()  # moi tinh huong bat dau sach
        t_start = time.perf_counter()
        first_alert, wrong_alert_s, last_t, raw_hits, raw_n = None, 0.0, None, 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                aborted = True
                break
            t = time.perf_counter()
            el = t - t_start
            if el >= PREP_S + ACT_S:
                break
            frame = cv2.flip(frame, 1)
            per = per_fn(frame)
            dec = classify(per["phones"], per["hands"], per["face"])
            raw = UNSURE if per["brightness"] < LOW_LIGHT else dec.state
            alert, _, _ = filt.update(t, raw)
            lat.append((time.perf_counter() - t) * 1000)

            acting = el >= PREP_S
            if acting:
                rows.append([rnd, code, f"{el - PREP_S:.3f}", raw, alert])
                if exp_raw is not None and el >= PREP_S + 1.0:  # bo 1 s dau (dang chuyen tu the)
                    raw_n += 1
                    raw_hits += raw == exp_raw
                if alert != NORMAL:
                    if alert == exp_alert and first_alert is None:
                        first_alert = el - PREP_S
                    if alert != exp_alert and last_t is not None:
                        wrong_alert_s += t - last_t
                last_t = t

            header = (f"[{idx + 1}/{len(plan)}] Vòng {rnd}: {text}", True)
            sub = (f"Chuẩn bị... {PREP_S - el:.0f}" if not acting else f"Đang đo: còn {PREP_S + ACT_S - el:.0f} s"
                   f"  ·  khung: {raw}  ·  cảnh báo: {alert}", False)
            cv2.imshow(win, overlay(frame, [header, sub], (255, 200, 80) if not acting else (120, 230, 140)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                aborted = True
                break
        if aborted:
            break
        trials.append(dict(round=rnd, code=code, text=text, exp_alert=exp_alert, first_alert=first_alert,
                           wrong_alert_s=wrong_alert_s, raw_acc=raw_hits / raw_n if raw_n else None))
    cap.release()
    cv2.destroyAllWindows()
    if not trials:
        raise SystemExit("Khong co du lieu.")

    with open(out_dir / f"frames_{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["vong", "tinh_huong", "t_s", "trang_thai_khung", "canh_bao"])
        w.writerows(rows)

    # Tong hop theo tinh huong
    by = defaultdict(list)
    for tr in trials:
        by[tr["code"]].append(tr)
    x = np.array(lat[10:]) if len(lat) > 20 else np.array(lat)
    fps = len(rows) / (len(trials) * ACT_S)
    md = [f"# Kết quả đo DriverGuard ({datetime.now():%Y-%m-%d %H:%M})", "",
          f"Máy: laptop CPU (webcam 640×480). {len(trials)} lượt = {len(by)} tình huống × "
          f"{max(t['round'] for t in trials)} vòng, mỗi lượt đo {ACT_S:.0f} s. Người thử tự đóng vai.", "",
          f"- Độ trễ ảnh → quyết định: **p50 {np.percentile(x, 50):.0f} ms, p95 {np.percentile(x, 95):.0f} ms**"
          f" · khoảng **{fps:.1f} FPS**", "",
          "| Tình huống | Cảnh báo mong đợi | Kết quả cảnh báo | Thời gian tới cảnh báo | Báo nhầm | Khung đúng |",
          "|---|---|---|---|---|---|"]
    for code, text, _, exp_alert in SCENARIOS:
        trs = by.get(code)
        if not trs:
            continue
        if exp_alert == NORMAL:
            ok = sum(t["first_alert"] is None and t["wrong_alert_s"] == 0 for t in trs)
            res, tta = f"{ok}/{len(trs)} lượt không báo", "–"
        else:
            hit = [t for t in trs if t["first_alert"] is not None]
            res = f"{len(hit)}/{len(trs)} lượt bắt được"
            tta = f"{np.mean([t['first_alert'] for t in hit]):.1f} s" if hit else "–"
        wrong = sum(t["wrong_alert_s"] for t in trs)
        accs = [t["raw_acc"] for t in trs if t["raw_acc"] is not None]
        acc = f"{np.mean(accs) * 100:.0f}%" if accs else "–"
        md.append(f"| {text} | {exp_alert} | {res} | {tta} | {wrong:.1f} s | {acc} |")
    counts = Counter((r[1], r[3]) for r in rows)
    md += ["", "Phân bố trạng thái từng khung (số khung):", "",
           "| Tình huống | " + " | ".join(s for s in (NORMAL, CALLING, TEXTING, SMOKING, UNSURE)) + " |",
           "|---|" + "---|" * 5]
    for code, text, _, _ in SCENARIOS:
        if code in by:
            md.append(f"| {code} | " + " | ".join(str(counts[(code, s)]) for s in (NORMAL, CALLING, TEXTING, SMOKING, UNSURE)) + " |")
    md += ["", "Ghi chú: \"Khung đúng\" bỏ 1 s đầu mỗi lượt (đang chuyển tư thế). \"Báo nhầm\" = số giây có cảnh báo"
           " sai loại hoặc cảnh báo khi không vi phạm.", f"Dữ liệu thô: `evidence/frames_{stamp}.csv`."]
    (out_dir / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    if aborted:
        print("\n(Da huy giua chung: chi tinh cac luot da xong.)")


if __name__ == "__main__":
    main()
