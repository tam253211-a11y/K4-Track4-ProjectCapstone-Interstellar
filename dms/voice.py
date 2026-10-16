"""Phat canh bao giong noi (mp3 tao san bang scripts/make_voice.py), khong chan vong lap camera.

Dung MCI cua Windows (winmm) nen khong can thu vien them. Khong co file hoac khong phai
Windows thi quay ve tieng bip.
"""
import ctypes
import shutil
import tempfile
import time
from pathlib import Path

from .rules import CALLING, SMOKING, TEXTING

SRC = Path(__file__).resolve().parent.parent / "assets" / "voice"
FILES = {CALLING: "calling.mp3", TEXTING: "texting.mp3", SMOKING: "smoking.mp3"}
REPEAT_S = 8.0  # canh bao van con thi nhac lai sau 8 s, khong doc lien tuc

try:
    _mci = ctypes.windll.winmm.mciSendStringW
except (AttributeError, OSError):
    _mci = None


class Voice:
    def __init__(self):
        # Chep sang thu muc tam (duong dan khong dau): MCI de loi voi duong dan co "Buổi"
        self.dir = Path(tempfile.gettempdir()) / "driverguard_voice"
        self.dir.mkdir(exist_ok=True)
        self.ok = _mci is not None and all((SRC / f).exists() for f in FILES.values())
        if self.ok:
            for f in FILES.values():
                shutil.copy(SRC / f, self.dir / f)
        self.last = {}

    def say(self, state):
        now = time.monotonic()
        if now - self.last.get(state, -1e9) < REPEAT_S:
            return
        self.last[state] = now
        if not self.ok:
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_ICONHAND)
            except ImportError:
                pass
            return
        _mci("close dg_alert", None, 0, None)
        _mci(f'open "{self.dir / FILES[state]}" type mpegvideo alias dg_alert', None, 0, None)
        _mci("play dg_alert", None, 0, None)  # khong co "wait" -> phat nen

    def close(self):
        if self.ok:
            _mci("close dg_alert", None, 0, None)
