# Auto Video

Auto Video chuyển lời thoại tiếng Trung thành **giọng đọc và phụ đề tiếng Việt** bằng Paraformer-zh, Gemini và TTS; hỗ trợ xử lý video dài, kiểm tra nội dung hội thoại, video dọc và ghép MP4.

## Cài trên máy Windows mới

Yêu cầu Windows 10/11 x64, Python 3.12, FFmpeg và Internet trong lần cài đặt đầu tiên.

```powershell
git clone https://github.com/lebao1462-cloud/auto_video.git
cd auto_video
git switch setup/windows-portable-20261016
.\setup_windows.bat
powershell -NoProfile -ExecutionPolicy Bypass -File .\check_portable_windows.ps1
.\start_auto_video.bat
```

Lần đầu mở ứng dụng, nhập **Gemini API key** trong giao diện. TTS Edge hoạt động qua Internet và không yêu cầu cài thêm VieNeu.

**Nếu muốn dùng VieNeu/giọng Ngọc Huyền:** chạy thêm `.\setup_vieneu_windows.bat` để cài thư viện `vieneu==3.8.3` và tải mô hình theo giấy phép tác giả.

Hướng dẫn và các hạn chế: [PORTABLE_WINDOWS.md](PORTABLE_WINDOWS.md).

Mã ứng dụng: [buzz-1.4.4](buzz-1.4.4/). Phiên bản Python và các thư viện được khai báo trong `buzz-1.4.4/pyproject.toml`, `buzz-1.4.4/uv.lock` và `buzz-1.4.4/localization-requirements.txt`.

**Không commit** API key, `.env`, tệp video, dữ liệu cá nhân, cache hoặc mô hình AI có dung lượng lớn. Mã nguồn trên GitHub không đồng nghĩa với EXE portable đã kiểm thử trên mọi máy; cần kiểm tra đầu ra với một video ngắn sau khi cài.
