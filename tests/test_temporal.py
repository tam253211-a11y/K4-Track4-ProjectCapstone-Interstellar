"""Kiem tra bo loc thoi gian bang thoi gian gia lap. Chay: python -m unittest discover -s tests -v"""
import unittest

from dms.rules import CALLING, NORMAL, TEXTING, UNSURE
from dms.temporal import AlertFilter


def run(seq, fps=10, f=None):
    """seq: list (so giay, trang thai). Tra ve list (t, alert) cho moi khung."""
    f = f or AlertFilter()
    out, t = [], 0.0
    for dur, state in seq:
        for _ in range(int(dur * fps)):
            out.append((t, f.update(t, state)[0]))
            t += 1 / fps
    return out


def first_alert(out, state):
    return next((t for t, a in out if a == state), None)


class TestAlertFilter(unittest.TestCase):
    def test_short_glance_does_not_alert(self):
        # Cam may len 0.5 s roi bo xuong -> khong canh bao
        out = run([(2, NORMAL), (0.5, CALLING), (3, NORMAL)])
        self.assertIsNone(first_alert(out, CALLING))

    def test_sustained_call_alerts_within_about_1_5s(self):
        out = run([(2, NORMAL), (4, CALLING)])
        t = first_alert(out, CALLING)
        self.assertIsNotNone(t)
        self.assertLess(t - 2.0, 1.6)

    def test_flicker_from_missed_detections_keeps_alert(self):
        # YOLO mat box 1 khung trong moi 4 khung -> van canh bao lien tuc
        seq = [(2, NORMAL)] + [(0.3, TEXTING), (0.1, NORMAL)] * 10
        out = run(seq)
        t0 = first_alert(out, TEXTING)
        self.assertIsNotNone(t0)
        self.assertTrue(all(a == TEXTING for t, a in out if t > t0 + 0.1))

    def test_alert_releases_after_calm(self):
        out = run([(4, CALLING), (3, NORMAL)])
        self.assertEqual(out[-1][1], NORMAL)
        # Van giu canh bao trong 0.5 s dau sau khi bo may xuong
        self.assertTrue(all(a == CALLING for t, a in out if 4.0 <= t < 4.5))

    def test_unsure_frames_do_not_count_as_violation(self):
        out = run([(5, UNSURE)])
        self.assertIsNone(first_alert(out, CALLING))
        self.assertIsNone(first_alert(out, TEXTING))

    def test_time_based_not_frame_based(self):
        # Cung 1.6 s goi, FPS khac nhau -> cung ket luan
        for fps in (6, 12, 25):
            out = run([(2, NORMAL), (1.6, CALLING)], fps=fps)
            self.assertIsNotNone(first_alert(out, CALLING), f"fps={fps}")


if __name__ == "__main__":
    unittest.main()
