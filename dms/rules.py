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

# Chi so moc ban tay MediaPipe (21 diem)
WRIST = 0
INDEX = (5, 6, 7, 8)     # goc, khop giua, khop dau, dau ngon tro
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)

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
MOUTH_NEAR = 0.6   # (YOLOE) tam thuoc la cach mieng < 0.6 do rong mat -> dang ngam
# Cu chi kep thuoc dua len mieng
GRIP_MOUTH_MAX = 0.45  # cho kep (giua ngon tro va ngon giua) cach mieng < 0.45 do rong mat
FINGER_GAP_MAX = 0.18  # ngon tro va ngon giua khep sat nhau (khop dau cach < 0.18)
EXTENDED = 1.05        # ngon duoi: dau ngon xa co tay hon khop giua it nhat 5%
RING_CURLED = 1.15     # ngon ap ut co: dau ngon khong xa co tay hon khop giua qua 15%

NORMAL, CALLING, TEXTING, SMOKING, UNSURE = "BINH THUONG", "DANG GOI", "DANG NHAN TIN", "DANG HUT THUOC", "KHONG CHAC"


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
    extra: dict = field(default_factory=dict)  # diem/so do de ve len camera


def point_box_dist(p, box):
    """Khoang cach tu diem toi hinh chu nhat (0 neu nam trong)."""
    x1, y1, x2, y2 = box[:4]
    dx = max(x1 - p[0], 0.0, p[0] - x2)
    dy = max(y1 - p[1], 0.0, p[1] - y2)
    return float(np.hypot(dx, dy))


def box_center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def hand_holds(hand, box, margin_px):
    """Tay dang cam dien thoai neu du nhieu moc tay nam trong box da noi rong."""
    x1, y1, x2, y2 = box[:4]
    inside = ((hand[:, 0] >= x1 - margin_px) & (hand[:, 0] <= x2 + margin_px)
              & (hand[:, 1] >= y1 - margin_px) & (hand[:, 1] <= y2 + margin_px))
    return int(inside.sum()) >= HAND_MIN_PTS


def classify(phones, hands, face, cigs=()):
    """phones, cigs: list (x1, y1, x2, y2, conf, ...); hands: list mang (21, 2); face: Face hoac None.

    Uu tien khi co nhieu hanh vi cung luc: goi > hut thuoc > nhan tin.
    """
    dec = classify_phone(phones, hands, face)
    if dec.state == CALLING:
        return dec
    # Tay dang cam dien thoai thi khong xet cu chi hut thuoc (cam may sat cam de bi nham)
    free_hands = hands
    if face is not None and phones:
        margin = HAND_MARGIN * face.width
        free_hands = [h for h in hands if not any(hand_holds(h, p, margin) for p in phones)]
    smoke = smoking_gesture(free_hands, face)
    if smoke.state != SMOKING and cigs:
        smoke = classify_smoking_box(cigs, hands, face)
    if smoke.state == SMOKING:
        return smoke
    # Giu so do cu chi tay-mieng de ve len camera ca khi chua du dieu kien hut thuoc
    for k in ("grip", "mouth", "mouth_d"):
        if k in smoke.extra:
            dec.extra[k] = smoke.extra[k]
    return dec


def smoking_gesture(hands, face):
    """Cu chi kep thuoc dua len mieng, chi tu moc tay va moc mat.

    Dieu thuoc nho va manh nen detector bo sot (YOLOE 4585 lop khong bat duoc, xem
    scripts/probe_smoking.py), nhung ban tay kep thuoc thi lon va on dinh:
    ngon tro + ngon giua duoi va khep sat, ngon ap ut co lai, cho kep o sat mieng.
    """
    if face is None or not hands:
        return Decision(NORMAL)
    best = None
    for h in hands:
        wrist = h[WRIST]
        reach = lambda i: float(np.linalg.norm(h[i] - wrist))  # noqa: E731
        extended = (reach(INDEX[3]) > reach(INDEX[1]) * EXTENDED
                    and reach(MIDDLE[3]) > reach(MIDDLE[1]) * EXTENDED)
        ring_curled = reach(RING[3]) < reach(RING[1]) * RING_CURLED
        gap = float(np.linalg.norm(h[INDEX[2]] - h[MIDDLE[2]])) / face.width
        grip = h[[INDEX[1], INDEX[2], MIDDLE[1], MIDDLE[2]]].mean(axis=0)  # cho kep dieu thuoc
        mouth_d = float(np.linalg.norm(grip - face.mouth)) / face.width
        ok = extended and ring_curled and gap < FINGER_GAP_MAX and mouth_d < GRIP_MOUTH_MAX
        if best is None or (ok, -mouth_d) > (best[0], -best[1]):
            best = (ok, mouth_d, gap, grip, extended, ring_curled)
    ok, mouth_d, gap, grip, extended, ring_curled = best
    extra = {"grip": grip, "mouth": face.mouth, "mouth_d": mouth_d}
    if ok:
        return Decision(SMOKING, reason=f"cử chỉ kẹp thuốc sát miệng ({mouth_d:.2f} < {GRIP_MOUTH_MAX})", extra=extra)
    return Decision(NORMAL, reason=f"tay cách miệng {mouth_d:.2f}", extra=extra)


def classify_smoking_box(cigs, hands, face):
    """Dung khi bat --yoloe: dua tren box thuoc la tu detector."""
    if face is None:
        return Decision(UNSURE, phone=max(cigs, key=lambda c: c[4]), reason="có thuốc lá nhưng không thấy mặt")
    margin = HAND_MARGIN * face.width
    best = None
    for c in cigs:
        mouth_d = float(np.linalg.norm(box_center(c) - face.mouth)) / face.width
        held = any(hand_holds(h, c, margin) for h in hands)
        if best is None or mouth_d < best[1]:
            best = (c, mouth_d, held)
    c, mouth_d, held = best
    extra = {"grip": box_center(c), "mouth": face.mouth, "mouth_d": mouth_d}
    # Ngam o mieng, hoac dang kep tren tay (giua hai hoi thuoc) deu tinh la hut thuoc
    if mouth_d < MOUTH_NEAR:
        return Decision(SMOKING, c, held, None, f"thuốc lá cách miệng {mouth_d:.2f} < {MOUTH_NEAR}", extra)
    if held:
        return Decision(SMOKING, c, held, None, f"tay kẹp thuốc lá, cách miệng {mouth_d:.2f}", extra)
    return Decision(NORMAL, c, held, None, f"thuốc lá xa miệng ({mouth_d:.2f}), không cầm", extra)


def classify_phone(phones, hands, face):
    if not phones:
        return Decision(NORMAL, reason="không thấy điện thoại")
    if face is None:
        # Khong co mat de lam moc thi khong do duoc khoang cach toi tai.
        return Decision(UNSURE, phone=max(phones, key=lambda p: p[4]), reason="có điện thoại nhưng không thấy mặt")

    margin = HAND_MARGIN * face.width
    best = None
    for p in phones:
        held = any(hand_holds(h, p, margin) for h in hands)
        da, db = point_box_dist(face.ear_a, p), point_box_dist(face.ear_b, p)
        ear = min(da, db) / face.width
        # Uu tien dien thoai dang duoc cam, roi den gan tai nhat
        key = (held, -ear)
        if best is None or key > best[0]:
            best = (key, p, held, ear, face.ear_a if da <= db else face.ear_b)
    _, p, held, ear, ear_pt = best
    c = box_center(p)
    u, v = face.to_face_coords(c)
    pw = (p[2] - p[0]) / face.width  # be rong hinh chieu cua dien thoai, don vi do rong mat
    extra = {"u": u, "v": v, "pw": pw, "ear_pt": ear_pt, "phone_c": c}
    geo = f"u={u:+.2f}, v={v:+.2f}, rộng={pw:.2f}"

    # Goi: gan tai VA nam ben canh mat VA ngang tam tai VA quay canh ve camera.
    # - Chi xet khoang cach thi dien thoai cam duoi cam cung "gan tai" -> bao nham.
    # - Cam ngang mat, mat lung quay vao camera thi box rong va trum len tai -> cung bao nham,
    #   nen can them be rong hinh chieu: ap tai thi chi thay canh dien thoai.
    if ear < EAR_NEAR and abs(u) >= SIDE_MIN and v <= EAR_BELOW_MAX and pw < CALL_WIDTH_MAX:
        return Decision(CALLING, p, held, ear, f"điện thoại sát tai, bên cạnh mặt ({geo})", extra)
    if held:
        return Decision(TEXTING, p, held, ear, f"tay cầm điện thoại ({geo})", extra)
    if abs(u) < FRONT_MAX and v > 0:
        # Tay hay bi dien thoai che nen MediaPipe mat moc tay; dien thoai ngay truoc mat/nguc van tinh
        return Decision(TEXTING, p, held, ear, f"điện thoại trước mặt/ngực ({geo})", extra)
    return Decision(NORMAL, p, held, ear, f"điện thoại để xa, không cầm (u={u:+.2f})", extra)
