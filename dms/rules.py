"""Luat hinh hoc: tu box dien thoai + moc tay + moc mat -> trang thai tai xe.

Moi khoang cach deu chia cho do rong mat (pixel), nen luat khong phu thuoc
viec tai xe ngoi gan hay xa camera.
"""
from dataclasses import dataclass, field

import numpy as np

# Chi so moc trong MediaPipe Face Landmarker (478 diem)
FACE_SIDE_A = 234  # mep mat ben trai anh, gan tai
FACE_SIDE_B = 454  # mep mat ben phai anh, gan tai
MOUTH = (13, 14)   # moi tren, moi duoi
CHIN = 152

# Nguong, tinh theo don vi "do rong mat"
EAR_NEAR = 0.45    # box dien thoai cach tai < 0.45 do rong mat -> dang ap tai
SIDE_MIN = 0.35    # khi goi, tam dien thoai lech ngang khoi tam mat it nhat 0.35
EAR_BELOW_MAX = 0.6  # khi goi, tam dien thoai khong thap hon tai qua 0.6
FRONT_MAX = 0.6    # |u| < 0.6: dien thoai o ngay truoc mat/nguc tai xe
# Ap tai thi dien thoai quay canh/nghieng ve camera -> box hep. Do tren webcam:
# ap tai 0.2-0.63, cam ngang mat lung quay vao camera ~1.1 -> chon 0.8 o giua.
CALL_WIDTH_MAX = 0.8
HAND_MARGIN = 0.30 # noi rong box dien thoai khi kiem tra tay co dang cam khong
HAND_MIN_PTS = 3   # so moc tay toi thieu nam trong box da noi rong

NORMAL, CALLING, TEXTING, UNSURE = "BINH THUONG", "DANG GOI", "DANG NHAN TIN", "KHONG CHAC"


@dataclass
class Face:
    ear_a: np.ndarray
    ear_b: np.ndarray
    mouth: np.ndarray
    chin: np.ndarray
    width: float  # do rong mat (pixel), don vi chuan hoa

    @classmethod
    def from_landmarks(cls, pts):
        """pts: mang (478, 2) toa do pixel."""
        a, b = pts[FACE_SIDE_A], pts[FACE_SIDE_B]
        return cls(a, b, (pts[MOUTH[0]] + pts[MOUTH[1]]) / 2, pts[CHIN], float(np.linalg.norm(a - b)))

    def to_face_coords(self, p):
        """Doi diem anh sang he toa do gan voi mat: goc giua hai tai, don vi do rong mat.
        u > 0: lech ve phia ear_b; v > 0: thap hon tai."""
        origin = (self.ear_a + self.ear_b) / 2
        x_axis = (self.ear_b - self.ear_a) / self.width
        y_axis = np.array([-x_axis[1], x_axis[0]])  # vuong goc, huong xuong cam
        d = np.asarray(p, dtype=float) - origin
        return float(d @ x_axis) / self.width, float(d @ y_axis) / self.width


@dataclass
class Decision:
    state: str
    phone: tuple | None = None   # (x1, y1, x2, y2, conf) cua dien thoai duoc chon
    held: bool = False
    ear_dist: float | None = None  # khoang cach toi tai gan nhat, don vi do rong mat
    reason: str = ""
    extra: dict = field(default_factory=dict)


def point_box_dist(p, box):
    """Khoang cach tu diem toi hinh chu nhat (0 neu nam trong)."""
    x1, y1, x2, y2 = box[:4]
    dx = max(x1 - p[0], 0.0, p[0] - x2)
    dy = max(y1 - p[1], 0.0, p[1] - y2)
    return float(np.hypot(dx, dy))


def hand_holds(hand, box, margin_px):
    """Tay dang cam dien thoai neu du nhieu moc tay nam trong box da noi rong."""
    x1, y1, x2, y2 = box[:4]
    inside = ((hand[:, 0] >= x1 - margin_px) & (hand[:, 0] <= x2 + margin_px)
              & (hand[:, 1] >= y1 - margin_px) & (hand[:, 1] <= y2 + margin_px))
    return int(inside.sum()) >= HAND_MIN_PTS


def classify(phones, hands, face):
    """phones: list (x1, y1, x2, y2, conf); hands: list mang (21, 2); face: Face hoac None."""
    if not phones:
        return Decision(NORMAL, reason="khong thay dien thoai")
    if face is None:
        # Khong co mat de lam moc thi khong do duoc khoang cach toi tai.
        return Decision(UNSURE, phone=max(phones, key=lambda p: p[4]), reason="co dien thoai nhung khong thay mat")

    margin = HAND_MARGIN * face.width
    best = None
    for p in phones:
        held = any(hand_holds(h, p, margin) for h in hands)
        ear = min(point_box_dist(face.ear_a, p), point_box_dist(face.ear_b, p)) / face.width
        # Uu tien dien thoai dang duoc cam, roi den gan tai nhat
        key = (held, -ear)
        if best is None or key > best[0]:
            best = (key, p, held, ear)
    _, p, held, ear = best
    u, v = face.to_face_coords(((p[0] + p[2]) / 2, (p[1] + p[3]) / 2))
    pw = (p[2] - p[0]) / face.width  # be rong hinh chieu cua dien thoai, don vi do rong mat
    extra = {"u": u, "v": v, "pw": pw}

    # Goi: gan tai VA nam ben canh mat VA ngang tam tai VA quay canh ve camera.
    # - Chi xet khoang cach thi dien thoai cam duoi cam cung "gan tai" -> bao nham.
    # - Cam ngang mat, mat lung quay vao camera thi box rong va trum len tai -> cung bao nham,
    #   nen can them be rong hinh chieu: ap tai thi chi thay canh dien thoai.
    if ear < EAR_NEAR and abs(u) >= SIDE_MIN and v <= EAR_BELOW_MAX and pw < CALL_WIDTH_MAX:
        return Decision(CALLING, p, held, ear, f"canh tai: u={u:+.2f}, v={v:+.2f}, rong={pw:.2f}", extra)
    if held:
        return Decision(TEXTING, p, held, ear, f"tay cam dien thoai: u={u:+.2f}, v={v:+.2f}, rong={pw:.2f}", extra)
    if abs(u) < FRONT_MAX and v > 0:
        # Tay hay bi dien thoai che nen MediaPipe mat moc tay; dien thoai ngay truoc mat/nguc van tinh
        return Decision(TEXTING, p, held, ear, f"truoc mat/nguc: u={u:+.2f}, v={v:+.2f}", extra)
    return Decision(NORMAL, p, held, ear, f"dien thoai de xa (gia do): u={u:+.2f}", extra)
