"""Kiem tra luat hinh hoc bang du lieu gia lap, khong can camera.

Chay: python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from dms.rules import CALLING, NORMAL, TEXTING, UNSURE, Face, classify


def make_face(cx=320, cy=200, width=160):
    """Mat gia: tai hai ben cach tam width/2, mieng va cam o duoi."""
    return Face(ear_a=np.array([cx - width / 2, cy]), ear_b=np.array([cx + width / 2, cy]),
                mouth=np.array([cx, cy + 50]), chin=np.array([cx, cy + 90]), width=float(width))


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

    def test_scale_invariant(self):
        # Cung bo cuc nhung tai xe ngoi xa hon (moi thu nho di mot nua) -> cung ket qua
        far = make_face(cx=160, cy=100, width=80)
        d = classify([(197, 75, 215, 130, 0.4)], [], far)
        self.assertEqual(d.state, CALLING)


if __name__ == "__main__":
    unittest.main()
