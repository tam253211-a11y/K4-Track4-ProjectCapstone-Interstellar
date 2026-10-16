# VinFast DriverGuard: cảnh báo tài xế dùng điện thoại và hút thuốc khi lái

Capstone Track 4 (Computer Vision & Robotics), Vin Perception Hackathon.
Thực hiện: Đặng Hữu Tâm (2A202602940), làm cá nhân.

Một webcam nhìn vào tài xế. Hệ thống nhận ra ba hành vi nguy hiểm: **gọi điện**, **dùng điện thoại** (nhắn tin, xem màn hình) và **hút thuốc**. Khi một hành vi kéo dài đủ lâu, hệ thống đọc cảnh báo bằng giọng nói tiếng Việt và ghi lại sự kiện kèm ảnh chụp cho quản lý đội xe.

## Bài toán và người dùng

- **Đơn vị Vin:** VinFast, hướng *giám sát tài xế*. Áp dụng được cho tài xế taxi Xanh SM và VinBus chạy ca dài.
- **Người dùng trực tiếp:** tài xế. Họ nghe cảnh báo bằng giọng nói ngay trong xe, không phải nhìn màn hình.
- **Người dùng thứ hai:** quản lý đội xe. Họ có thống kê vi phạm theo ca, ảnh chụp từng lần vi phạm và file CSV để xem lại hoặc nhắc nhở tài xế.
- **Nỗi đau:** chỉ vài giây không nhìn đường đã đủ gây tai nạn. Hiện nay quản lý đội xe không biết tài xế dùng điện thoại hay hút thuốc trong xe, trừ khi khách khiếu nại.

## Chạy thử

Yêu cầu: Windows, Python 3.10+, webcam. Không cần GPU.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Model được tải về thư mục `models/` (không đưa lên git):

```powershell
mkdir models
curl.exe -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl.exe -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
cd models; python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"; cd ..
```

| Lệnh | Việc |
|---|---|
| `python run.py` | Dashboard trong cửa sổ OpenCV (nhấn `q` để thoát) |
| `python run.py --web` | Dashboard trên trình duyệt: http://localhost:8000 (Ctrl+C để dừng) |
| `python run.py --yoloe` | Bật thêm YOLOE tìm khung thuốc lá (chậm, xem phần ca lỗi) |
| `python scripts/evaluate.py` | Đo có hướng dẫn trên màn hình, ghi ra `evidence/results.md` |
| `python -m unittest discover -s tests -v` | 25 unit test cho luật hình học và bộ lọc thời gian |

## Hệ thống

```
webcam 640x480 ──► YOLO11n (lớp "cell phone", COCO) ──────────┐
               ├─► MediaPipe Hand Landmarker (21 mốc/tay) ────┼─► luật hình học trong hệ toạ độ khuôn mặt
               └─► MediaPipe Face Landmarker (tai, miệng) ────┘        (dms/rules.py)
                                                                          │ trạng thái từng khung
                                                                          ▼
                    bộ lọc thời gian ≥60% trong 2 s (dms/temporal.py) ──► cảnh báo
                                                                          │
                     giọng nói · dashboard · ảnh chụp + CSV ◄─────────────┘
```

| File | Vai trò |
|---|---|
| `dms/perception.py` | Chạy YOLO + MediaPipe, đổi mốc sang pixel, nhớ vị trí mặt 1,5 s |
| `dms/rules.py` | Luật hình học: gọi / dùng điện thoại / hút thuốc / không chắc |
| `dms/temporal.py` | Bộ lọc theo thời gian, tách cảnh báo khỏi nhiễu từng khung |
| `dms/voice.py`, `assets/voice/` | Giọng cảnh báo tiếng Việt (tạo sẵn bằng edge-tts, phát offline) |
| `dms/dashboard.py`, `dms/events.py` | Dashboard cho quản lý đội xe, nhật ký vi phạm |
| `dms/webserver.py` | Phát dashboard lên localhost (MJPEG, chỉ nghe 127.0.0.1) |

## Trụ cột Track 4 và vị trí trong hệ thống

**1. Hình học** (`dms/rules.py`, `Face.to_face_coords`)
- Mọi khoảng cách đều chia cho **độ rộng mặt**, nên tài xế ngồi gần hay xa camera vẫn cho cùng kết quả (test `test_scale_invariant`).
- Vị trí điện thoại được đổi sang **hệ toạ độ gắn với khuôn mặt**: `u` là độ lệch ngang so với tâm mặt, `v` là độ lệch dọc so với tầm tai. Trục ngang chạy theo đường nối hai tai, nên vẫn đúng khi tài xế nghiêng đầu.
- Dùng **bề rộng hình chiếu** của điện thoại: khi áp lên tai, điện thoại quay cạnh về phía camera nên khung hẹp.
- Trên camera, đường vàng nối điện thoại tới tai và đường tím nối ngón tay tới miệng, để người xem thấy luật đang đo gì.

Vì sao chọn luật hình học thay vì train một bộ phân loại: 2 giờ không đủ dữ liệu có nhãn. Luật hình học giải thích được từng quyết định ("điện thoại cách tai 0,32 lần độ rộng mặt"), và mỗi ngưỡng đều gắn với một ca lỗi thật (bảng dưới).

**2. Biểu diễn**
Tài xế được biểu diễn bằng mốc tai và miệng (từ 478 mốc mặt) cộng 21 mốc mỗi bàn tay. Điện thoại là một khung box. Đầu ra là một trong năm trạng thái rời rạc. **Hút thuốc được nhận ra qua tư thế bàn tay**: ngón trỏ và ngón giữa duỗi và khép sát, ngón áp út co, chỗ kẹp nằm sát miệng. Không cần tìm thấy điếu thuốc.

**3. Thời gian** (`dms/temporal.py`)
- Chỉ cảnh báo khi hành vi chiếm **≥ 60% thời gian trong 2 giây gần nhất**, tính theo giây chứ không theo số khung, vì FPS dao động. Liếc điện thoại 0,5 giây không bị báo. YOLO mất khung 1/4 số lần thì cảnh báo vẫn liên tục (6 unit test).
- Tắt cảnh báo sau 1 giây trở lại bình thường. Giọng nói nhắc lại tối đa mỗi 8 giây.
- Đo **độ trễ từ lúc có ảnh tới lúc ra quyết định** theo p50/p95, hiển thị trực tiếp trên dashboard.

**4. Bất định**
- Không thấy mặt, hoặc ảnh tối hơn 40/255, thì hệ thống báo **"không chắc"** thay vì đoán. Khung "không chắc" không bị tính là vi phạm, cũng không bị tính là an toàn.
- Điện thoại che mặt thì dùng vị trí mặt gần nhất trong 1,5 giây.
- Hạ ngưỡng YOLO xuống 0,25 vì điện thoại áp tai chỉ đạt độ tin cậy khoảng 0,37. Báo nhầm được chặn bằng hình học và bộ lọc thời gian.

**Học** (YOLO11n pretrained trên COCO, MediaPipe pretrained) và **Dữ liệu** (nhật ký CSV và ảnh chụp mỗi vi phạm, `evidence/frames_*.csv`) đóng vai trò bổ trợ.

## Bằng chứng

### Số đo

Đo trên laptop Ryzen 5 5500U, chạy CPU (GPU GTX 1650 không dùng được do driver 2020 chỉ hỗ trợ CUDA 11.1), webcam 640×480:

| Cấu hình | Độ trễ ảnh → quyết định | FPS |
|---|---|---|
| YOLO11n (imgsz 480) + MediaPipe tay + mặt, kèm dashboard | p50 74 ms, p95 88 ms | ~12 |
| Thêm YOLOE tìm thuốc lá cùng tiến trình | p50 185 ms | ~5 |
| YOLOE ở tiến trình riêng, 2 nhân CPU | p50 ~100 ms, p95 ~190 ms | ~8–10 |

Tỉ lệ phát hiện và báo nhầm theo từng tình huống (gọi, dùng điện thoại, liếc nhanh, hút thuốc, che miệng khi ngáp) được đo bằng `python scripts/evaluate.py`. Kết quả nằm trong [`evidence/results.md`](evidence/results.md) sau khi chạy.

### Ca lỗi gặp thật và cách xử lý

| # | Ca lỗi | Nguyên nhân | Xử lý |
|---|---|---|---|
| 1 | Cầm điện thoại dưới cằm bị báo **đang gọi** | Luật chỉ xét khoảng cách tới tai (0,30 < 0,45). Cạnh trên khung ngang tầm tai. | Thêm hướng: khi gọi, điện thoại phải nằm **bên cạnh** mặt (\|u\| ≥ 0,35) |
| 2 | Cầm ngang mặt, mặt lưng quay vào camera, bị báo **đang gọi** | Khung trùm lên tai (khoảng cách 0) và lệch sang bên (u = −0,65) | Thêm bề rộng hình chiếu: áp tai thì khung hẹp (0,2–0,63), cầm ngang mặt thì khung rộng (~1,1). Ngưỡng 0,8. |
| 3 | Điện thoại áp tai độ tin cậy chỉ 0,37 | Quay cạnh về phía camera, YOLO chỉ thấy một dải mỏng | Hạ ngưỡng xuống 0,25, chặn báo nhầm bằng hình học và thời gian |
| 4 | **YOLOE (4.585 lớp, có lớp "cigarette") không thấy điếu thuốc** | 36 lần chạy vùng miệng, ngưỡng 0,03: không lần nào ra `cigarette`. Model gán nhãn cả cảnh (ear, nose, manicure…), vật nhỏ và mảnh bị lấn át. Chạy CPU tốn 240–380 ms mỗi lần. | Chuyển sang **nhận diện cử chỉ** từ mốc tay. YOLOE giữ sau cờ `--yoloe`. Kiểm chứng lại bằng `scripts/probe_smoking.py`. |
| 5 | MediaPipe crash khi khởi động | Đường dẫn có dấu tiếng Việt (`Buổi` bị đọc thành `Bu?i`) | Nạp model bằng bytes thay vì đường dẫn |

### Giới hạn (những gì chưa chứng minh)

- Chỉ thử trên **một người**, một webcam laptop đặt chính diện, trong phòng đủ sáng. Chưa thử trong xe thật, camera đặt góc, ban đêm, hay người đeo kính râm.
- Hút thuốc được nhận ra qua **cử chỉ**, nên **dễ nhầm** với ăn, uống hay cắn móng tay bằng hai ngón. Ngược lại, kẹp thuốc bằng tư thế khác thì sẽ bỏ sót. Hệ thống không thật sự nhìn thấy điếu thuốc.
- Ngưỡng hình học được chỉnh trên vài lần thử của chính người làm, chưa có tập kiểm tra độc lập.
- Khoảng cách hoạt động ước chừng 0,5–1,5 m, mặt phải hiện trong khung hình.

## Tác động nếu triển khai ở Vin (ước lượng)

- **Cơ sở:** WHO (2011, *Mobile phone use: a growing problem of driver distraction*) cho biết tài xế dùng điện thoại có nguy cơ gặp tai nạn **cao khoảng 4 lần**.
- **Cách ước lượng cho một đội xe:** số xe × số giờ lái mỗi ngày × tỉ lệ thời gian dùng điện thoại × mức giảm nhờ cảnh báo. Các số của đội xe Xanh SM hoặc VinBus cần lấy từ dữ liệu vận hành thật. README không tự đặt số.
- **Chi phí:** phần mềm chạy được trên CPU laptop phổ thông (khoảng 12 FPS), không cần GPU. Có thể dùng camera cabin sẵn có.
- **Đường mở rộng:** camera hồng ngoại cho ban đêm, chạy trên thiết bị nhúng trong xe, gửi sự kiện về trung tâm điều hành, train bộ phân loại cử chỉ khi đã có dữ liệu được gán nhãn.
- **Rủi ro:** quyền riêng tư của tài xế (cần đồng ý, chỉ lưu ảnh khi vi phạm). Báo nhầm nhiều gây khó chịu và tài xế sẽ tắt hệ thống. Không được dùng làm căn cứ kỷ luật duy nhất.

## Dữ liệu an toàn

Mọi ảnh và video đều do chính người làm tự đóng vai trước webcam. Dùng bút thay thuốc lá thật. Không dùng dữ liệu của người khác.

---
`index.html`: trang đề bài hackathon (gốc của repo này).
