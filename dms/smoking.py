"""Phat hien thuoc la bang YOLOE prompt-free (tu vung 4585 lop, co 'cigarette', 'cigar', 'pen').

YOLOE ton ~250-380 ms/lan tren CPU nen chay o TIEN TRINH rieng, ~2 lan/giay, voi 2 nhan CPU;
vong lap chinh (dien thoai, tay, mat) giu 4 nhan con lai. Chay chung mot tien trinh thi hai
model gianh nhau nhan CPU va vong lap chinh tut tu ~12 xuong ~5 FPS.
Hut mot hoi keo dai vai giay nen 2 Hz la du.
Khi biet vi tri mat, chi dua vung quanh mieng vao model: thuoc la la vat nho, cat vung
thi no to hon trong anh dau vao.
"""
import multiprocessing as mp
import queue
import time
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
CIG_CLASSES = {922: "cigar", 924: "cigarette"}
PEN_CLASSES = {2990: "pen", 2992: "pencil"}  # theo doi rieng de phan tich bao nham
CROP_SCALE = 1.6    # canh vung cat = 1.6 lan do rong mat, tam o mieng
STALE_S = 1.5       # ket qua cu hon 1.5 s thi bo
MIN_PERIOD_S = 0.5  # toi da ~2 lan/giay
WORKER_THREADS = 2  # so nhan CPU cho YOLOE


def _crop(frame, mouth, fw):
    h, w = frame.shape[:2]
    if mouth is None or not fw:
        return 0, 0, frame
    half = int(CROP_SCALE * fw / 2)
    x1, y1 = max(int(mouth[0]) - half, 0), max(int(mouth[1]) - half, 0)
    x2, y2 = min(int(mouth[0]) + half, w), min(int(mouth[1]) + half, h)
    if x2 - x1 <= 32 or y2 - y1 <= 32:
        return 0, 0, frame
    return x1, y1, frame[y1:y2, x1:x2]


def _worker(jobs, results, imgsz, conf):
    import torch
    torch.set_num_threads(WORKER_THREADS)
    from ultralytics import YOLOE

    model = YOLOE(str(MODELS / "yoloe-11s-seg-pf.pt"))
    classes = list(CIG_CLASSES) + list(PEN_CLASSES)
    while True:
        job = jobs.get()
        if job is None:
            return
        frame, mouth, fw = job
        ox, oy, img = _crop(frame, mouth, fw)
        a = time.perf_counter()
        r = model.predict(img, imgsz=imgsz, conf=conf, classes=classes, verbose=False)[0]
        ms = (time.perf_counter() - a) * 1000
        cigs, pens = [], []
        for (bx1, by1, bx2, by2), c, k in zip(r.boxes.xyxy.cpu().numpy().tolist(),
                                              r.boxes.conf.cpu().numpy().tolist(),
                                              r.boxes.cls.cpu().numpy().astype(int).tolist()):
            item = (bx1 + ox, by1 + oy, bx2 + ox, by2 + oy, c, CIG_CLASSES.get(k) or PEN_CLASSES.get(k))
            (cigs if k in CIG_CLASSES else pens).append(item)
        results.put((cigs, pens, ms))


class SmokingDetector:
    def __init__(self, imgsz=320, conf=0.2):
        ctx = mp.get_context("spawn")
        self.jobs, self.results = ctx.Queue(maxsize=1), ctx.Queue()
        self.proc = ctx.Process(target=_worker, args=(self.jobs, self.results, imgsz, conf), daemon=True)
        self.proc.start()
        self.result_t, self.cigs, self.pens = 0.0, [], []
        self.infer_ms = 0.0
        self.last_submit = 0.0

    def submit(self, frame, mouth=None, face_width=None):
        now = time.perf_counter()
        if now - self.last_submit < MIN_PERIOD_S or self.jobs.full():
            return  # tien trinh con dang ban: bo khung nay, khong xep hang
        self.last_submit = now
        try:
            self.jobs.put_nowait((frame, None if mouth is None else tuple(mouth), face_width))
        except queue.Full:
            pass

    def latest(self):
        """Tra ve (thuoc la, but); moi phan tu (x1, y1, x2, y2, conf, ten) trong toa do khung hinh."""
        try:
            while True:
                self.cigs, self.pens, self.infer_ms = self.results.get_nowait()
                self.result_t = time.perf_counter()
        except queue.Empty:
            pass
        if time.perf_counter() - self.result_t < STALE_S:
            return self.cigs, self.pens
        return [], []

    def close(self):
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass
        self.proc.join(timeout=2)
        if self.proc.is_alive():
            self.proc.terminate()
