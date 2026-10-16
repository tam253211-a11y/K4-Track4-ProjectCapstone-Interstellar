"""Ghi su kien vi pham: dem theo ca truc, anh chup luc vi pham, file CSV de quan ly doi xe xem lai."""
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from .rules import CALLING, NORMAL, SMOKING, TEXTING

LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "events"
KINDS = (CALLING, TEXTING, SMOKING)
SLUG = {CALLING: "goi", TEXTING: "nhantin", SMOKING: "hutthuoc"}


@dataclass
class Event:
    kind: str
    start: datetime
    t0: float
    duration: float | None  # None khi su kien con dang dien ra
    thumb: object            # anh nho cho dashboard
    snapshot: str


class EventLog:
    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.csv_path = LOG_DIR / f"events_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["bat_dau", "loai", "keo_dai_s", "anh"])
        self.events = []
        self.counts = {k: 0 for k in KINDS}
        self.seconds = {k: 0.0 for k in KINDS}
        self.current = None
        self.session_start = datetime.now()

    def update(self, t, alert, frame):
        """t: perf_counter; alert: trang thai da loc; frame: anh da ve (de chup)."""
        if self.current and alert != self.current.kind:
            self._close(t)
        if alert in KINDS and self.current is None:
            now = datetime.now()
            path = LOG_DIR / f"{now:%Y%m%d_%H%M%S}_{SLUG[alert]}.jpg"
            cv2.imwrite(str(path), frame)
            self.current = Event(alert, now, t, None, cv2.resize(frame, (96, 72)), path.name)
            self.events.append(self.current)
            self.counts[alert] += 1
        if self.current:
            self.current.duration = t - self.current.t0

    def _close(self, t):
        ev = self.current
        ev.duration = t - ev.t0
        self.seconds[ev.kind] += ev.duration
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([f"{ev.start:%Y-%m-%d %H:%M:%S}", ev.kind, f"{ev.duration:.1f}", ev.snapshot])
        self.current = None

    def total_seconds(self, kind):
        live = self.current.duration if self.current and self.current.kind == kind else 0.0
        return self.seconds[kind] + live

    def close(self, t):
        if self.current:
            self._close(t)
