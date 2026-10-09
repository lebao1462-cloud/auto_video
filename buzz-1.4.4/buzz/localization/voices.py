"""Voice catalogue shared by the Auto Video localization UIs."""

EDGE_VOICE_CHOICES = (
    ("Edge · Hoài My · Nữ", "vi-VN-HoaiMyNeural"),
    ("Edge · Nam Minh · Nam", "vi-VN-NamMinhNeural"),
)

# The preset IDs below are the exact names returned by VieNeu-TTS v3 Turbo 3.8.3.
_VIENEU = (
    ("Adam bựa", "Nam · Bắc · tự nhiên"),
    ("Trúc Ly", "Nữ · Bắc · tự nhiên"),
    ("Thiện Minh", "Nam · Bắc · kể chuyện"),
    ("Mai Anh", "Nữ · Bắc · tin tức"),
    ("Hải Đăng", "Nam · Bắc · tự nhiên"),
    ("Thùy Dung", "Nữ · Nam · tin tức"),
    ("Thiền Tâm Đức", "Nam · Bắc · kể chuyện"),
    ("Ngọc Huyền", "Nữ · Bắc · tự nhiên"),
    ("Quang Sơn", "Nam · Trung · tự nhiên"),
    ("Ngọc Trân", "Nữ · Trung · tự nhiên"),
    ("Minh Đức", "Nam · Bắc · tin tức"),
    ("Phạm Tuyên", "Nam · Bắc · tự nhiên"),
    ("Thái Sơn", "Nam · Nam · kể chuyện"),
    ("Xuân Vĩnh", "Nam · Bắc · tự nhiên"),
    ("Thanh Bình", "Nam · Bắc · kể chuyện"),
    ("Ngọc Linh", "Nữ · Bắc · kể chuyện"),
    ("Đoan Trang", "Nữ · Bắc · tự nhiên"),
    ("Thục Đoan", "Nữ · Nam · kể chuyện"),
    ("Minh Triết", "Nam · Nam · tin tức"),
    ("Mỹ Duyên", "Nữ · Nam · đọc truyện"),
    ("Quỳnh Anh", "Nữ · Bắc · đọc truyện"),
    ("Đức Trí", "Nam · Nam · đọc truyện"),
    ("Kim Thanh", "Nữ · Nam · đọc truyện"),
    ("Adam", "Nam · Nam · tự nhiên"),
    ("Quốc Tuấn", "Nam · Bắc · tự nhiên"),
)

VIENEU_VOICE_CHOICES = tuple(
    (f"VieNeu · {name} · {description}", f"vieneu:{name}")
    for name, description in _VIENEU
)
TTS_VOICE_CHOICES = EDGE_VOICE_CHOICES + VIENEU_VOICE_CHOICES


def parse_tts_voice(value: str | None) -> tuple[str, str]:
    """Return provider and voice while accepting legacy Edge values."""
    value = value or "vi-VN-HoaiMyNeural"
    if value.startswith("vieneu:"):
        return "vieneu", value.split(":", 1)[1]
    if value.startswith("edge:"):
        return "edge", value.split(":", 1)[1]
    return "edge", value
