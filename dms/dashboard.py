"""Dashboard cho quan ly doi xe, ve trong mot cua so OpenCV (khong dung Streamlit: stream webcam
tren Windows hay giat). Moi o tren man hinh deu la du lieu that cua phien chay nay.

Chu tieng Viet co dau duoc ve bang Pillow (cv2.putText khong ve duoc dau).
"""
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .events import KINDS
from .rules import CALLING, NORMAL, SMOKING, TEXTING, UNSURE
from .temporal import RATIO

W, H = 1280, 720
CAM = (12, 64, 820, 615)  # x, y, rong, cao vung camera
PX = 846                  # cot ben phai
PW = W - PX - 12

# Mau RGB
BG, CARD, LINE = (14, 20, 30), (24, 32, 46), (44, 56, 76)
TEXT, MUTED, ACCENT = (232, 238, 246), (140, 156, 178), (90, 160, 255)
STATE_RGB = {NORMAL: (34, 170, 90), CALLING: (230, 60, 60), TEXTING: (245, 140, 30),
             SMOKING: (170, 80, 220), UNSURE: (230, 190, 40)}
LABEL = {NORMAL: "Bình thường", CALLING: "Đang gọi điện thoại", TEXTING: "Đang dùng điện thoại",
         SMOKING: "Đang hút thuốc", UNSURE: "Không chắc chắn"}
SHORT = {CALLING: "Gọi điện", TEXTING: "Nhắn tin / xem", SMOKING: "Hút thuốc"}

FONT_DIR = "C:/Windows/Fonts/"


def _font(name, size):
    try:
        return ImageFont.truetype(FONT_DIR + name, size)
    except OSError:
        return ImageFont.load_default()


F = {s: _font("segoeui.ttf", s) for s in (12, 13, 14, 16, 20)}
FB = {s: _font("segoeuib.ttf", s) for s in (13, 14, 16, 18, 20, 22, 28, 32)}
BANNER_H = 64  # banner canh bao o day camera, doc duoc tu xa (may chieu)


def bgr(rgb):
    return rgb[::-1]


PANEL_PERIOD_S = 0.2  # ve lai chu/panel 5 lan/giay; camera van cap nhat moi khung
# Ve chu bang Pillow tren ca anh 1280x720 moi khung lam FPS tut tu ~10 xuong ~6.


class Dashboard:
    def __init__(self):
        self.texts = []
        self.cache, self.cache_t = None, -1e9
        self.banners = {}

    def banner(self, alert):
        """Ve san banner cho moi loai canh bao mot lan (Pillow cham neu ve moi khung)."""
        if alert not in self.banners:
            cw = CAM[2]
            pil = Image.new("RGB", (cw, BANNER_H), STATE_RGB[alert])
            ImageDraw.Draw(pil).text((cw // 2, BANNER_H // 2), "CẢNH BÁO: " + LABEL[alert].upper(),
                                     font=FB[32], fill=(255, 255, 255), anchor="mm")
            self.banners[alert] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return self.banners[alert]

    def render(self, cam, alert, raw, best, ratio, reason, events, stats):
        now = time.perf_counter()
        if self.cache is None or now - self.cache_t >= PANEL_PERIOD_S:
            self.cache, self.cache_t = self._render_full(cam, alert, raw, best, ratio, reason, events, stats), now
            x, y = CAM[:2]
            self.cam_label = self.cache[y:y + 26, x:x + 230].copy()
        img = self.cache.copy()
        x, y, cw, ch = CAM
        img[y:y + ch, x:x + cw] = cv2.resize(cam, (cw, ch))
        img[y:y + 26, x:x + 230] = self.cam_label
        if alert in (CALLING, TEXTING, SMOKING):
            img[y + ch - BANNER_H:y + ch, x:x + cw] = self.banner(alert)
        # Vien camera nhap nhay khi canh bao (ve moi khung de nhay deu)
        blink = int(now * 2) % 2 == 0
        border = STATE_RGB[alert] if alert != NORMAL else LINE
        thick = 6 if alert != NORMAL and blink else 2
        cv2.rectangle(img, (x - 1, y - 1), (x + cw, y + ch), bgr(border), thick)
        return img

    def t(self, x, y, s, font, color=TEXT, anchor="la"):
        self.texts.append((x, y, s, font, color, anchor))

    def card(self, img, x, y, w, h, title=None):
        cv2.rectangle(img, (x, y), (x + w, y + h), bgr(CARD), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), bgr(LINE), 1)
        if title:
            self.t(x + 12, y + 8, title.upper(), FB[13], MUTED)

    def _render_full(self, cam, alert, raw, best, ratio, reason, events, stats):
        self.texts = []
        img = np.full((H, W, 3), bgr(BG), np.uint8)

        # Thanh tieu de
        cv2.rectangle(img, (0, 0), (W, 52), bgr(CARD), -1)
        cv2.line(img, (0, 52), (W, 52), bgr(LINE), 1)
        self.t(16, 10, "VinFast DriverGuard", FB[22])
        self.t(258, 17, "Giám sát tài xế · điện thoại & thuốc lá", F[14], MUTED)
        dur = datetime.now() - events.session_start
        self.t(W - 16, 10, f"{datetime.now():%H:%M:%S}", FB[18], anchor="ra")
        self.t(W - 16, 32, f"Ca trực {str(dur).split('.')[0]}", F[12], MUTED, anchor="ra")

        # Camera
        x, y, cw, ch = CAM
        cv2.rectangle(img, (x, y), (x + 230, y + 26), bgr((0, 0, 0)), -1)
        self.t(x + 8, y + 4, "● CAM 01 · Cabin tài xế (webcam)", F[13], (240, 90, 90))

        # Trang thai hien tai
        py = 64
        col = STATE_RGB[alert if alert != NORMAL else (UNSURE if raw == UNSURE else NORMAL)]
        self.card(img, PX, py, PW, 118, "Trạng thái hiện tại")
        cv2.rectangle(img, (PX + 12, py + 32), (PX + PW - 12, py + 74), bgr(col), -1)
        head = ("CẢNH BÁO: " + LABEL[alert].upper()) if alert != NORMAL else LABEL[UNSURE if raw == UNSURE else NORMAL].upper()
        self.t(PX + PW // 2, py + 53, head, FB[22] if len(head) < 28 else FB[18], (255, 255, 255), anchor="mm")
        bx, by, bw = PX + 12, py + 84, PW - 24
        cv2.rectangle(img, (bx, by), (bx + bw, by + 8), bgr(LINE), -1)
        cv2.rectangle(img, (bx, by), (bx + int(bw * min(ratio, 1)), by + 8), bgr(STATE_RGB[best]), -1)
        mx = bx + int(bw * RATIO)
        cv2.line(img, (mx, by - 3), (mx, by + 11), bgr(TEXT), 2)
        self.t(PX + 12, py + 96, f"{SHORT.get(best, '')} {ratio * 100:.0f}% trong 2 s (ngưỡng {RATIO * 100:.0f}%)",
               F[12], MUTED)

        # Thong ke ca truc
        py = 192
        self.card(img, PX, py, PW, 112, "Thống kê vi phạm ca trực")
        cw3 = (PW - 24 - 16) // 3
        for i, k in enumerate(KINDS):
            cx = PX + 12 + i * (cw3 + 8)
            cv2.rectangle(img, (cx, py + 32), (cx + cw3, py + 102), bgr(BG), -1)
            cv2.line(img, (cx, py + 32), (cx + cw3, py + 32), bgr(STATE_RGB[k]), 3)
            self.t(cx + 10, py + 38, SHORT[k], F[13], MUTED)
            self.t(cx + 10, py + 54, str(events.counts[k]), FB[28], STATE_RGB[k] if events.counts[k] else TEXT)
            self.t(cx + 10 + 22 * len(str(events.counts[k])), py + 70, "lần", F[12], MUTED)
            self.t(cx + cw3 - 8, py + 84, f"{events.total_seconds(k):.0f} s", F[12], MUTED, anchor="ra")

        # Dong su kien: lap day khoang giua thong ke va suc khoe he thong
        py = 314
        health_h = 108
        health_y = H - 44 - health_h
        n_show = (health_y - 10 - py - 30) // 64
        self.card(img, PX, py, PW, health_y - 10 - py, "Dòng sự kiện")
        recent = events.events[-n_show:][::-1]
        if not recent:
            self.t(PX + 12, py + 40, "Chưa có vi phạm trong ca này.", F[14], MUTED)
        for i, ev in enumerate(recent):
            ey = py + 30 + i * 64
            img[ey:ey + 56, PX + 12:PX + 12 + 75] = cv2.resize(ev.thumb, (75, 56))
            cv2.rectangle(img, (PX + 12, ey), (PX + 87, ey + 56), bgr(STATE_RGB[ev.kind]), 2)
            self.t(PX + 98, ey + 2, LABEL[ev.kind], FB[14], STATE_RGB[ev.kind])
            live = ev is events.current
            self.t(PX + 98, ey + 22, f"{ev.start:%H:%M:%S} · kéo dài {ev.duration:.1f} s" + (" · đang diễn ra" if live else ""),
                   F[13], TEXT)
            self.t(PX + 98, ey + 38, f"Ảnh: logs/events/{ev.snapshot}", F[12], MUTED)

        # Suc khoe he thong
        py = health_y
        self.card(img, PX, py, PW, health_h, "Sức khỏe hệ thống")
        rows = [("FPS", f"{stats['fps']:.1f}"),
                ("Độ trễ ảnh → quyết định", f"p50 {stats['p50']:.0f} ms · p95 {stats['p95']:.0f} ms"),
                ("Độ sáng / mặt", f"{stats['brightness']:.0f}/255 · {stats['face']}"),
                ("Khung hiện tại", LABEL[raw].lower())]
        for i, (k, v) in enumerate(rows):
            self.t(PX + 12, py + 30 + i * 19, k, F[13], MUTED)
            self.t(PX + PW - 12, py + 30 + i * 19, v, F[13], anchor="ra")

        # Chan trang
        self.t(12, H - 28, f"Lý do: {reason}", F[13], MUTED)
        self.t(W - 12, H - 28, "Bản demo: 1 webcam laptop · dữ liệu do người thử tự đóng vai · nhấn q để thoát",
               F[12], MUTED, anchor="ra")

        # Ve toan bo chu mot lan bang Pillow
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for tx, ty, s, font, color, anchor in self.texts:
            d.text((tx, ty), s, font=font, fill=color, anchor=anchor)
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
