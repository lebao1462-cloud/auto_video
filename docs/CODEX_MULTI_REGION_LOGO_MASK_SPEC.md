# Đặc tả cho Codex — NÂNG CẤP bộ lọc CHE LOGO hiện có (video ngang + dọc)

**Phạm vi đã được người dùng xác nhận:** Chỉ sửa chức năng che logo/watermark nguồn đã có của Auto Video để có thể chọn nhiều góc, kéo thả, chỉnh từng vùng tùy ý cho cả 16:9 và 9:16. **KHÔNG sửa bất cứ chức năng phụ đề nào.** Đây là đặc tả, chưa triển khai.

## 1. Hiện trạng phải giữ tương thích

Căn cứ vào mã đã kiểm tra trên nhánh `setup/windows-portable-20261016`:
- `buzz-1.4.4/buzz/widgets/localization_dashboard.py`: checkbox `blur_bilibili_watermark_checkbox` (mặc định bật), `LocalizationJob`, lựa chọn chiều video thủ công, tối đa 5 jobs và nút thêm video.
- `buzz-1.4.4/buzz/widgets/localization_dialog.py`: `LocalizationWorker` truyền `blur_bilibili_watermark` và `portrait_subtitle_overlay` vào `FinalRenderOptions`.
- `buzz-1.4.4/buzz/localization/final_render.py`: `FinalRenderOptions`, `build_bilibili_watermark_filter_graph` để che logo cố định với video ngang; `build_portrait_subtitle_filter_graph` cho video dọc, trong đó phần che logo góc trên/dưới đang đi chung đồ thị với lớp xử lý phụ đề.
- `buzz-1.4.4/buzz/localization/workflow.py` và `chunked.py`: pipeline, checkpoint, render từng part.

**Không viết lại pipeline, không thay GUI xử lý video, không đụng ASR/Gemini/TTS/Ben Review/subtitle.** Khi chưa bật chế độ mới, kết quả phải giống bản cũ.

## 2. Giao diện — chỉ chỉnh che logo

Thêm ngay bên cạnh checkbox che logo Bilibili hiện tại một nút `Tùy chỉnh vùng che logo...` mở PyQt6 `QDialog` với:
1. Chế độ `Mặc định (như hiện tại)` và `Tùy chỉnh`, mặc định legacy để giữ nguyên hành vi người dùng hiện có.
2. Bốn checkbox độc lập: **Trên trái / Trên phải / Dưới trái / Dưới phải**. Có thể chọn cả bốn đồng thời. Tick một góc thêm vùng che mặc định tương ứng; bỏ tick loại vùng che preset đó, không ảnh hưởng vùng khác.
3. Khung xem trước một frame thật từ video đang chọn (nếu nhiều video, người dùng chọn video đích). Vùng che hiển thị là hình chữ nhật có thể **kéo để di chuyển, kéo cạnh và 4 góc để resize**, có danh sách chọn vùng / thêm / xóa / bật tắt; nút `+ Thêm vùng tự do` tạo vùng ở vị trí bất kỳ, không bó buộc vào bốn góc.
4. Mỗi vùng có: tên, `x/y/w/h`, `blur_radius`, bật/tắt. **Giữ nguyên kiểu làm mờ Blur hiện có**; chưa cần thêm Mosaic, AI xoá vật thể, phủ màu hay bất kỳ chức năng nào ngoài yêu cầu.
5. Slider độ mờ, spinbox tọa độ và kích thước (%) cập nhật cùng overlay. Có thể lưu/tải preset riêng cho video ngang và dọc, copy cấu hình sang job khác khi người dùng chọn rõ ràng.
6. `Xem gốc / Xem sau khi che`, nút `Render thử 5 giây`, hủy preview, áp dụng, hủy, khôi phục mặc định.
7. FFprobe/FFmpeg trích thumbnail, render thử chạy trong thread/worker, không khóa giao diện và không gọi Gemini/TTS.

Phải xử lý hiển thị đúng khi resize cửa sổ, letterbox, high-DPI và video xoay 90°; vị trí chọn trên UI phải khớp frame thực tế sau khi encode.

## 3. Schema và trạng thái

Đề xuất `buzz/localization/logo_mask.py` (không phụ thuộc Qt), dataclass bất biến `LogoMaskConfig` chứa `schema_version: 1`, `mode: legacy|custom`, `regions: tuple[LogoMaskRegion,...]`. Mỗi vùng có id duy nhất, name, enabled, x, y, w, h, blur_radius.

Ví dụ:
```json
{
  "schema_version": 1,
  "mode": "custom",
  "regions": [
    {"id":"top-left", "name":"Trên trái", "enabled":true, "x":0.015, "y":0.02, "w":0.20, "h":0.10, "blur_radius":24},
    {"id":"bottom-right", "name":"Dưới phải", "enabled":true, "x":0.79, "y":0.85, "w":0.19, "h":0.10, "blur_radius":22}
  ]
}
```

- Toạ độ chuẩn hoá [0,1] theo hình người dùng nhìn thấy; rectangle phải có kích thước dương, x+w ≤ 1 và y+h ≤ 1, blur radius 1–40; từ chối NaN/Infinity, id trùng và hơn 16 vùng đang bật.
- `mode=legacy`: gọi y nguyên bộ lọc logo cũ ở các trường hợp tương ứng, bất kể `regions`.
- `mode=custom, regions=[]`: không che logo; tuyệt đối không áp dụng chồng vùng logo legacy.
- Lưu JSON trong output/checkpoint riêng của video và optional preset; không chứa API key, đường dẫn riêng trong mẫu preset. Chụp snapshot bất biến khi video được thêm vào hàng đợi; sửa giao diện trong lúc xử lý không làm đổi video đang chạy.
- Retry/resume phải dùng cùng cấu hình; fingerprint cấu hình chỉ ảnh hưởng render, không xoá checkpoint ASR/Gemini/TTS khi thay vùng che và render lại.

## 4. Tích hợp FFmpeg — thay đúng bộ lọc logo, không đụng phụ đề

- Tạo builder che logo đa vùng dựa trên bộ lọc cũ: một đồ thị `filter_complex` xử lý N vùng trong **một lượt encode** bằng `split/crop/boxblur/overlay`, label riêng cho từng vùng. Hạn chế crop tối thiểu 2px, làm tròn/clamp theo frame thực, không thay đổi tỷ lệ video.
- Trong `FinalRenderOptions` chỉ bổ sung tùy chọn mask bất biến. Giữ `blur_bilibili_watermark`/tham số legacy để các lời gọi hiện có không thay đổi.
- Với video ngang: thay bộ lọc che một góc hiện tại bằng vùng che tùy chỉnh **chỉ khi mode=custom**; cách burn phụ đề Việt, che phụ đề Trung và chèn Ben Review giữ nguyên.
- Với video dọc: TÁCH hợp lý nhánh xử lý *logo góc trên/dưới* ra khỏi `build_portrait_subtitle_filter_graph` để thay bằng các mask tùy chỉnh; NHÁNH che phụ đề Trung và đốt phụ đề Việt giữ nguyên thứ tự, kích thước, vị trí. Nếu `mode=legacy`, kết quả phải như cũ.
- Trình tự logic bắt buộc: **video gốc → bộ lọc logo (legacy/custom) → lớp xử lý phụ đề Trung hiện tại → phụ đề Việt hiện tại → logo Ben Review hiện tại → MP4**. Bộ lọc logo không được áp dụng sau phụ đề vì sẽ làm mờ chữ Việt hay logo Ben Review.
- Giữ nguyên `cover_original_subtitles`, `portrait_subtitle_overlay`, `_portrait_one_line_ass`, kích cỡ font, vùng phụ đề dọc ~68–71%, phụ đề ngang 2 dòng, đồng bộ TTS, freeze frame/timing, logo Ben Review và output.
- Áp dụng trong render `soft/burn/none` nhưng KHÔNG thay đổi semantics của từng chế độ. Nếu không dùng mask và không cần filter khác thì phải giữ stream-copy. Không đụng đến âm thanh.
- Dùng list args cho FFmpeg và validate toàn bộ giá trị để tránh injection. Cùng builder cho preview 5s và MP4 final để kết quả khớp.

## 5. Test chống hồi quy

- Unit: chọn 1, 2, 3, 4 góc; 1–16 vùng; custom rỗng; kéo đến sát mép; thay blur cho từng vùng; nhiều rectangle trùng nhau; ratio và pixel cho 16:9/9:16/4:3; rotation; JSON import invalid; job snapshot; retry/resume.
- GUI PyQt6: checkbox các góc độc lập, drag/resize, viewport letterbox, xóa vùng tự do, áp dụng/hủy, lưu preset, nhiều job có config riêng; preview async/hủy không treo.
- FFmpeg synthetic clips: 2/4 góc bị làm mờ, trung tâm không ảnh hưởng; video ngang và dọc, audio không mất, duration không đổi, sample tại đầu/giữa/cuối; preview 5s đúng cùng mask.
- **Regression cứng:** test vị trí/phông/kích thước và hai dòng/1 dòng phụ đề cũ, che phụ đề Trung, Gemini/TTS timing, Ben Review vẫn chạy xanh. Không xóa hoặc nới điều kiện test phụ đề để qua test.
- Mở lại project cũ checkpoint thiếu trường mask -> chạy legacy như trước. Chạy 5 job song song không đọc nhầm config.

## 6. Tiêu chí hoàn thành và cách làm

1. Tạo nhánh feature code riêng xuất phát từ `setup/windows-portable-20261016`; không thay đổi `master`.
2. Đọc mã/đánh dấu các nhánh legacy của logo và phụ đề trước khi refactor; thêm model+validation+test trước.
3. Làm builder FFmpeg đa vùng; test trên clip tổng hợp với âm thanh. Thêm editor PyQt6 và nút dashboard. Tích hợp job snapshot, checkpoint và preview.
4. Chạy toàn bộ kiểm thử cũ và mới; xác minh bản video dọc phụ đề Việt nguyên vẹn và Ben Review không bị mờ.
5. Tạo PR với screenshot giao diện và video render thử; báo chính xác số test passed/skipped, file đổi và hạn chế. Không commit API key, model ONNX, video gốc hay MP4 đầu ra.

**Không tự động đưa code vào chạy sản xuất trước khi xác minh. Không thực hiện điều chỉnh phụ đề ngoài phạm vi này.**
