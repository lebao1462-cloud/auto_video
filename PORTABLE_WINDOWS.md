# Auto Video — cài đặt trên máy Windows khác

> **Trạng thái:** Tài liệu và công cụ kiểm tra này áp dụng cho mã nguồn **đang có trên GitHub**. Bản Auto Video mới nhất đã chỉnh sửa trên máy `DESKTOP-L9P8PI4` trong ngày 15–16/10/2026 **chưa được đồng bộ** vì máy đó đang mất kết nối điều khiển. **Không coi nhánh này là bản phát hành mới nhất** cho tới khi hoàn thành đồng bộ mã nguồn và kiểm thử trên máy sạch.

## 1. Yêu cầu

- Windows 10/11 **64 bit**, Python **3.12.x** 64 bit.
- Git, Internet để cài thư viện và tải mô hình AI lần đầu.
- FFmpeg và FFprobe trong `PATH` (bộ cài `setup_windows.bat` sẽ thử cài bằng winget).
- Khuyến nghị RAM >= 16 GB và ổ trống đủ cho mô hình + dữ liệu render. Không cần chép cache hoặc video từ máy cũ để cài chương trình.

## 2. Clone, cài, mở

Mở **PowerShell** tại thư mục bạn muốn đặt Auto Video:

```powershell
git clone https://github.com/lebao1462-cloud/auto_video.git
cd auto_video
git switch setup/windows-portable-20261016
.\setup_windows.bat
.\check_portable_windows.ps1
.\start_auto_video.bat
```

Nhánh `setup/windows-portable-20261016` hiện mới bao gồm phần chuẩn bị cài đặt; nhánh `master` chưa đồng bộ bản mã nguồn mới nhất từ máy gốc. Nếu chính sách Windows chặn PowerShell script, thay bằng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\check_portable_windows.ps1
```

`setup_windows.bat` cài ứng dụng ở `.venv` riêng thông qua `buzz-1.4.4/pyproject.toml` và `buzz-1.4.4/localization-requirements.txt`. Các phiên bản được khai báo trong `poetry.lock` và `uv.lock` (nếu chọn quy trình tương ứng). **Không** copy nguyên `.venv` từ máy cũ: virtualenv có đường dẫn phụ thuộc từng máy.

## 3. Gemini và nhận dạng tiếng Trung

- Mở chương trình và nhập **Gemini API key trong giao diện** của bạn. Tuyệt đối không đưa key vào GitHub, `.bat`, tài liệu, log công khai hay commit.
- Paraformer-zh (FunASR/ModelScope) tải các mô hình khi sử dụng lần đầu. Thiết lập `MODELSCOPE_CACHE` nếu muốn cất mô hình ở ổ khác.
- `FFmpeg`/`FFprobe` là chương trình hệ thống, không phải thư viện Python; phải có trên máy mới.

## 4. VieNeu TTS (giọng Ngọc Huyền)

Auto Video dùng một **Python riêng biệt** để chạy VieNeu ONNX. Mã nguồn GUI trên GitHub **không bao gồm** bản phát hành VieNeu, mô hình ONNX và cache Hugging Face. Cài VieNeu theo hướng dẫn **chính thức của nhà cung cấp/giấy phép của mô hình**, trong một môi trường riêng, sau đó cấu hình **trên máy mới** các biến:

```powershell
$env:BUZZ_VIENEU_PYTHON = "C:\AutoVideo-VieNeu\.venv\Scripts\python.exe"
$env:BUZZ_VIENEU_MODEL_DIR = "C:\AIModels\VieNeu-TTS-v3-Turbo\onnx_update"
$env:BUZZ_VIENEU_HF_HOME = "C:\AIModels\HuggingFace"
```

Ba đường dẫn ở trên chỉ là **ví dụ**, không phải thư mục được tạo tự động. Thay bằng đường dẫn có thật và đảm bảo môi trường VieNeu import được module `vieneu`. Chạy `start_auto_video.bat` **trong cùng phiên PowerShell** để giữ các biến; nếu muốn cấu hình vĩnh viễn, khai báo trong Windows Environment Variables (không ghi vào repository).

Những giọng, các file tham chiếu và bộ mô hình cần kiểm tra riêng: không thể đảm bảo giọng `Ngọc Huyền` hoạt động chỉ bằng cách `git clone` và chạy `pip install`.

## 5. Không đưa lên GitHub

- API keys/tokens, mật khẩu, file `.env`, database người dùng và cấu hình riêng.
- Video nguồn/đầu ra `.mp4`, `.wav`, phụ đề cá nhân, checkpoint dịch, log, cache, bộ mô hình `.onnx`, `.pt`, `.safetensors`.
- Thư mục `.venv`, `build`, `dist`, cache pip/uv, bản chạy PyInstaller build trên từng máy.

GitHub lưu **mã nguồn + phiên bản thư viện khai báo + hướng dẫn**, không nhất thiết chứa toàn bộ file nhị phân; các trọng số mô hình phải được tải hoặc chuyển riêng theo giấy phép.

## 6. Kiểm tra

Chạy `check_portable_windows.ps1` để thấy thư viện, FFmpeg và đường dẫn VieNeu đang thiếu. Sau khi thiết lập xong, thử chạy một video ngắn với Gemini rồi kiểm tra MP4 có hình, tiếng và phụ đề trước khi xử lý phim dài.

**Chưa xác nhận kiểm thử trên máy Windows sạch**. Cần đồng bộ mã nguồn mới nhất từ `D:\codex\auto_video`, soát key và kiểm thử lại trước khi đánh dấu bản portable hoàn chỉnh.
