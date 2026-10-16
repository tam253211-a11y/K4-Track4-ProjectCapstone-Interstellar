"""Kiem tra luat hinh hoc bang du lieu gia lap, khong can camera.

Chay: python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from dms.rules import CALLING, NORMAL, SMOKING, TEXTING, UNSURE, Face, classify


def make_face(cx=320, cy=200, width=160):
    """Mat gia: tai hai ben cach tam width/2, mieng va cam o duoi."""
    return Face(ear_a=np.array([cx - width / 2, cy]), ear_b=np.array([cx + width / 2, cy]),
                mouth=np.array([cx, cy + 50]), chin=np.array([cx, cy + 90]), width=float(width))


def make_smoke_hand(dx=0.0, dy=0.0, ring_open=False, spread_fingers=False):
    """Tay kep thuoc: ngon tro + giua duoi, khep sat, cho kep ngay canh mieng (320, 250).
    Mat gia rong 160 px. dx, dy dich ca ban tay (pixel)."""
    pts = np.array([
        (380, 400),                                         # 0 co tay
        (365, 385), (350, 365), (340, 345), (335, 330),     # 1-4 ngon cai
        (360, 330), (348, 290), (338, 262), (330, 240),     # 5-8 ngon tro
        (372, 328), (360, 288), (350, 262), (343, 240),     # 9-12 ngon giua
        (385, 330), (378, 305), (372, 320), (376, 338),     # 13-16 ngon ap ut (co)
        (397, 338), (392, 320), (388, 332), (390, 345),     # 17-20 ngon ut (co)
    ], dtype=float)
    if ring_open:  # ban tay mo (che mieng khi ngap): ngon ap ut duoi thang
        pts[14], pts[15], pts[16] = (380, 290), (378, 262), (376, 240)
    if spread_fingers:  # chu V: ngon tro va ngon giua xoe xa nhau
        pts[[7, 8]] -= (45, 0)
        pts[[11, 12]] += (45, 0)
    return pts + (dx, dy)


def make_hand(cx, cy, spread=20):
    rng = np.random.default_rng(0)
    return np.array([cx, cy]) + rng.uniform(-spread, spread, size=(21, 2))


class TestClassify(unittest.TestCase):
    def test_no_phone_is_normal(self):
        self.assertEqual(classify([], [], make_face()).state, NORMAL)

    def test_phone_at_ear_is_calling(self):
        face = make_face()  # tai phai o x=400, y=200
        phone = (395, 150, 430, 260, 0.4)
        d = classify([phone], [make_hand(420, 230)], face)
        self.assertEqual(d.state, CALLING)
        self.assertLess(d.ear_dist, 0.45)

    def test_phone_tilted_at_ear_is_calling(self):
        # Ap tai, dien thoai nghieng nen box rong ~0.63 do rong mat (do tren webcam)
        face = make_face()  # tai trai o x=240, y=200
        phone = (150, 150, 250, 260, 0.72)  # rong 100/160 = 0.625
        d = classify([phone], [make_hand(200, 220)], face)
        self.assertEqual(d.state, CALLING)

    def test_calling_does_not_need_visible_hand(self):
        d = classify([(395, 150, 430, 260, 0.4)], [], make_face())
        self.assertEqual(d.state, CALLING)

    def test_phone_in_hand_below_face_is_texting(self):
        face = make_face()
        phone = (280, 330, 360, 430, 0.8)  # truoc nguc
        d = classify([phone], [make_hand(320, 400)], face)
        self.assertEqual(d.state, TEXTING)
        self.assertTrue(d.held)

    def test_phone_under_chin_is_texting_not_calling(self):
        # Loi da gap tren webcam: dien thoai ngay duoi cam cach tai < 0.45 nhung nam giua mat
        face = make_face()  # tai o y=200, do rong 160
        phone = (285, 250, 350, 400, 0.86)
        d = classify([phone], [], face)
        self.assertLess(d.ear_dist, 0.45)
        self.assertEqual(d.state, TEXTING)

    def test_phone_back_facing_camera_beside_face_is_not_calling(self):
        # Loi thu 2 tren webcam: cam ngang mat, lech trai, mat lung quay vao camera.
        # Box trum len tai (cach tai = 0), lech ngang u ~ -0.65 nhung rong ~1.1 do rong mat.
        face = make_face()  # tai trai o x=240
        phone = (150, 130, 325, 300, 0.89)
        d = classify([phone], [make_hand(250, 290)], face)
        self.assertEqual(d.ear_dist, 0.0)
        self.assertEqual(d.state, TEXTING)

    def test_phone_without_hand_is_normal(self):
        # Dien thoai tren gia do, khong ai cam
        d = classify([(500, 380, 580, 470, 0.9)], [make_hand(150, 400)], make_face())
        self.assertEqual(d.state, NORMAL)

    def test_phone_without_face_is_unsure(self):
        self.assertEqual(classify([(300, 300, 360, 400, 0.8)], [], None).state, UNSURE)

    def test_cigarette_at_mouth_is_smoking(self):
        face = make_face()  # mieng o (320, 250)
        d = classify([], [], face, cigs=[(310, 245, 360, 258, 0.4, "cigarette")])
        self.assertEqual(d.state, SMOKING)

    def test_cigarette_held_away_from_mouth_is_smoking(self):
        face = make_face()
        cig = (430, 330, 470, 342, 0.35, "cigarette")
        d = classify([], [make_hand(450, 336)], face, cigs=[cig])
        self.assertEqual(d.state, SMOKING)

    def test_cigarette_lying_far_without_hand_is_normal(self):
        d = classify([], [], make_face(), cigs=[(560, 440, 600, 452, 0.3, "cigarette")])
        self.assertEqual(d.state, NORMAL)

    def test_calling_beats_smoking_and_smoking_beats_texting(self):
        face = make_face()
        cig = (310, 245, 360, 258, 0.4, "cigarette")
        calling = classify([(395, 150, 430, 260, 0.4)], [], face, cigs=[cig])
        self.assertEqual(calling.state, CALLING)
        texting = classify([(280, 330, 360, 430, 0.8)], [make_hand(320, 400)], face, cigs=[cig])
        self.assertEqual(texting.state, SMOKING)

    def test_grip_gesture_at_mouth_is_smoking(self):
        self.assertEqual(classify([], [make_smoke_hand()], make_face()).state, SMOKING)

    def test_grip_gesture_away_from_mouth_is_normal(self):
        # Cung tu the tay nhung ha xuong nguc (cach mieng ~1 do rong mat)
        self.assertEqual(classify([], [make_smoke_hand(dy=160)], make_face()).state, NORMAL)

    def test_open_hand_at_mouth_is_not_smoking(self):
        # Che mieng khi ngap: ca ban tay mo -> ngon ap ut khong co
        self.assertEqual(classify([], [make_smoke_hand(ring_open=True)], make_face()).state, NORMAL)

    def test_v_sign_at_mouth_is_not_smoking(self):
        self.assertEqual(classify([], [make_smoke_hand(spread_fingers=True)], make_face()).state, NORMAL)

    def test_hand_holding_phone_at_chin_is_texting_not_smoking(self):
        hand = make_smoke_hand()
        phone = (300, 220, 400, 360, 0.8)  # tay nay dang cam dien thoai
        self.assertEqual(classify([phone], [hand], make_face()).state, TEXTING)

    def test_scale_invariant(self):
        # Cung bo cuc nhung tai xe ngoi xa hon (moi thu nho di mot nua) -> cung ket qua
        far = make_face(cx=160, cy=100, width=80)
        d = classify([(197, 75, 215, 130, 0.4)], [], far)
        self.assertEqual(d.state, CALLING)


if __name__ == "__main__":
    unittest.main()
