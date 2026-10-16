# Auto Video — cài trên máy Windows khác

Mã nguồn Auto Video được đồng bộ theo nhánh `setup/windows-portable-20261016` từ máy phát triển Windows. Các công cụ cài đặt, kiểm tra thư viện và mô hình VieNeu được giữ trong cùng repository. **Chưa xác minh cài đặt từ đầu trên một máy Windows khác.**

## Yêu cầu

- Windows 10/11 64 bit; Python **3.12.x** (64 bit).
- Git và Internet để cài thư viện, lấy mô hình lúc sử dụng lần đầu.
- FFmpeg/FFprobe trong PATH (script cài đặt có thể thử cài bằng winget).
- Khuyến nghị RAM từ 16 GB và dung lượng trống tối thiểu vài GB; mô hình, video và dữ liệu render có thể cần nhiều hơn.
- Không cần sao chép thư mục `.venv`, cache, video hoặc khóa API từ máy cũ.

## Bước 1. Clone và cài Auto Video

Mở PowerShell trong thư mục bạn muốn sử dụng:

```powershell
git clone https://github.com/lebao1462-cloud/auto_video.git
cd auto_video
git switch setup/windows-portable-20261016
.\setup_windows.bat
powershell -NoProfile -ExecutionPolicy Bypass -File .\check_portable_windows.ps1
.\start_auto_video.bat
```

Khi mở chương trình, nhập Gemini API key vào giao diện trên **máy mới**. Khóa không được lưu trong GitHub. `setup_windows.bat` sẽ tạo `.venv` rồi cài thư viện ứng dụng theo `buzz-1.4.4/pyproject.toml` và `buzz-1.4.4/localization-requirements.txt`. Repository có `uv.lock` và `poetry.lock` cho các trường hợp cần đồng bộ thư viện theo các trình quản lý đó.

Nếu bạn chỉ dùng **Gemini + Edge TTS**, có thể bỏ qua cài VieNeu dưới đây. Các mô hình Paraformer-zh qua FunASR/ModelScope được tải về cache khi sử dụng lần đầu; lần tải đầu tiên cần Internet.

## Bước 2 (tùy chọn). Cài VieNeu cho giọng Ngọc Huyền

Mô hình ONNX của VieNeu quá lớn để đưa vào Git. Script dưới đây cài `vieneu==3.8.3` trong **môi trường Python riêng**, tải các trọng số VieNeu và bộ giải mã MOSS từ các repository của tác giả trên Hugging Face:

```powershell
.\setup_vieneu_windows.bat
powershell -NoProfile -ExecutionPolicy Bypass -File .\check_portable_windows.ps1 -RequireVieNeu
.\start_auto_video.bat
```

- VieNeu: <https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo>
- Bộ giải mã âm thanh: <https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX>
- Thư viện VieNeu: phiên bản `3.8.3`, chạy ONNX/CPU độc lập với môi trường ứng dụng.

Dữ liệu mặc định trên máy mới, tương đối so với thư mục clone:

```text
auto_video/
  .venv/                           (môi trường GUI)
  .vieneu-venv/                    (môi trường VieNeu)
  models/
    VieNeu-TTS-v3-Turbo/onnx_update/
    HuggingFace/hub/
```

`start_auto_video.bat` nhận biết những đường dẫn này khi có file tương ứng, và truyền các biến `BUZZ_VIENEU_PYTHON`, `BUZZ_VIENEU_MODEL_DIR`, `BUZZ_VIENEU_HF_HOME` cho chương trình. Bạn cũng có thể tự cấu hình ba biến trên trỏ đến các mô hình đã tải ở vị trí khác; script sẽ không ghi đè biến đã đặt. VieNeu và mô hình là tài sản của bên thứ ba: tôn trọng giấy phép và quyền tải của tác giả.

Nếu chỉ muốn cài thư viện trước, chưa tải mô hình: `setup_vieneu_windows.bat --no-models`. Khi cần tải sau, chạy `.vieneu-venv\Scripts\python.exe install_vieneu_models.py`.

## Bước 3. Kiểm tra và dùng

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\check_portable_windows.ps1
.\start_auto_video.bat
```

Chọn một video ngắn, dịch bằng Gemini, xuất MP4 rồi kiểm tra hình, âm thanh, phụ đề và giọng đọc. Các bản ghi checkpoint và video dài không được sao chép theo repository. Chương trình có thể tự tải mô hình Paraformer khi dùng, cần mạng và đủ dung lượng đĩa.

Nếu bạn muốn xây bản chạy độc lập .exe không cần cài Python trên máy đích thì đó là **bước đóng gói riêng**; mã nguồn cùng lockfile/bộ cài Python không tương đương với một bản portable EXE đã kiểm thử.

## Không đưa lên GitHub

Không commit khóa Gemini, `.env`, token, mật khẩu, log cá nhân, video nguồn/đầu ra, checkpoint người dùng, cache `.venv`/uv/pip, tệp mô hình `.onnx`, `.pt`, `.safetensors` hoặc các thư mục chứa dữ liệu cá nhân. Kiểm tra `git status` trước mọi lần push.

## Kiểm thử

Các bài kiểm thử đơn vị của dự án nằm ở `buzz-1.4.4/tests`. Chạy ví dụ:

```powershell
cd buzz-1.4.4
..\.venv\Scripts\python.exe -m pytest -q tests/localization
```

Việc có đủ thư viện **không đảm bảo** kết quả dựng video trên mọi loại CPU/GPU. Cần một lượt kiểm thử end-to-end trên máy Windows mới để xác nhận chính thức.
