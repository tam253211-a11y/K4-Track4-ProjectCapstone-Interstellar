"""Bo loc theo thoi gian: bien trang thai tung khung hinh thanh canh bao on dinh.

Chi bat canh bao khi mot hanh vi chiem >= RATIO thoi gian trong WINDOW_S giay gan nhat
(giong PERCLOS cho mat nham). Tinh theo giay chu khong theo so khung, vi FPS dao dong.
"""
from collections import deque

from .rules import CALLING, NORMAL, SMOKING, TEXTING, UNSURE

WINDOW_S = 2.0   # cua so quan sat
RATIO = 0.6      # hanh vi phai chiem >= 60% thoi gian trong cua so
RELEASE_S = 1.0  # tro lai binh thuong lien tuc 1 s moi tat canh bao
MIN_SPAN_S = 1.0 # cua so phai co it nhat 1 s du lieu that (khong canh bao ngay luc khoi dong)

ALERTS = (CALLING, TEXTING, SMOKING)


class AlertFilter:
    def __init__(self, window_s=WINDOW_S, ratio=RATIO, release_s=RELEASE_S):
        self.window_s, self.ratio, self.release_s = window_s, ratio, release_s
        self.hist = deque()  # (t, state)
        self.alert = NORMAL
        self.calm_since = None

    def fractions(self, now):
        """Ti le thoi gian moi trang thai chiem trong cua so, co trong so theo do dai moi khung."""
        span = {s: 0.0 for s in (NORMAL, *ALERTS, UNSURE)}
        items = list(self.hist)
        for (t, s), (t_next, _) in zip(items, items[1:] + [(now, None)]):
            span[s] += max(t_next - t, 0.0)
        total = sum(span.values())
        return {s: v / total for s, v in span.items()} if total > 0 else span, total

    def update(self, t, raw_state):
        self.hist.append((t, raw_state))
        while self.hist and self.hist[0][0] < t - self.window_s:
            self.hist.popleft()
        frac, total = self.fractions(t)

        # Khung KHONG CHAC bi loai khoi mau so: khong tinh la vi pham, cung khong tinh la an toan
        sure = total * (1 - frac[UNSURE])
        best = max(ALERTS, key=lambda s: frac[s])
        best_ratio = frac[best] * total / sure if sure > 0 else 0.0

        if total >= MIN_SPAN_S and best_ratio >= self.ratio:
            self.alert, self.calm_since = best, None
        elif self.alert in ALERTS:
            if raw_state == NORMAL:
                self.calm_since = self.calm_since if self.calm_since is not None else t
                if t - self.calm_since >= self.release_s:
                    self.alert, self.calm_since = NORMAL, None
            else:
                self.calm_since = None
        return self.alert, best, best_ratio
