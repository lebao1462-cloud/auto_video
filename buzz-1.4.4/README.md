# Auto Video

Công cụ trên máy tính để chuyển video có lời thoại tiếng Trung hoặc tiếng Anh
thành MP4 có giọng đọc và phụ đề tiếng Việt. Mã nguồn của dự án đặt tại
[lebao1462-cloud/auto_video](https://github.com/lebao1462-cloud/auto_video).

## Chức năng

1. Nhận dạng lời thoại và mốc thời gian bằng Whisper.
2. Với Gemini, sửa lỗi nhận dạng tiếng Trung theo ngữ cảnh rồi dịch theo nhóm tối đa 30 đoạn có ID và thời gian; Argos vẫn là lựa chọn ngoại tuyến.
3. Tạo giọng Việt bằng Edge TTS và ghép theo các mốc của video.
4. Có thể tách lời thoại gốc bằng Demucs để giữ nhạc và tiếng nền.
5. Tạo SRT/VTT và xuất MP4 với phụ đề gắn vào hình hoặc phụ đề rời.
6. Tự chia video dài thành phần 15 phút, xử lý lần lượt và ghép thành một MP4.

Chất lượng nhận dạng và bản dịch tự động cần được kiểm tra trước khi đăng tải.
Đặc biệt, bản dịch Trung → Anh → Việt của Argos có thể sai nghĩa.

## Chạy trên máy Windows đã cài đặt
```powershell
powershell -ExecutionPolicy Bypass -File D:\Dev\run-auto-video.ps1
```

Mã nguồn và môi trường Python ở `D:\codex\auto_video`. Phần cài đặt,
mô hình và tệp đầu ra trên máy này được đặt trên ổ D.

Từ giao diện, chọn **File → Localize Video to Vietnamese...**, chọn video
và thư mục xuất rồi bấm Start. Máy này có sẵn `small.pt` tại
`D:\Dev\buzz-models\whisper\small.pt`; để dùng Gemini cần chọn Gemini
và cung cấp API key. Sau khi hoàn tất, kiểm tra MP4 và file phụ đề cạnh nhau.
Ví dụ đã thử: `D:\codex\auto_video_output\mayhutbui_v2\mayhutbui.vi.mp4`.

## Tài liệu và mã nguồn

- [Hướng dẫn xử lý video](LOCALIZATION_GUIDE.md)
- [Giấy phép và thông báo bản quyền](LICENSE)

Một số tên thư mục và module nội bộ vẫn giữ nguyên để các mô hình, dữ liệu,
thiết lập và bản ghi trước đây tiếp tục hoạt động. Giao diện và Help chỉ hiển thị các chức năng, hướng dẫn sử dụng của Auto Video.
