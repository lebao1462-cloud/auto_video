import pytest

from buzz.argos_worker import _resolve_route


class FakeTranslation:
    def __init__(self, prefix):
        self.prefix = prefix

    def translate(self, text):
        return self.prefix + text


class FakeLanguage:
    def __init__(self, code):
        self.code = code
        self.routes = {}

    def get_translation(self, target):
        return self.routes.get(target.code)


def test_worker_prefers_direct_chinese_vietnamese_route():
    zh = FakeLanguage("zh")
    en = FakeLanguage("en")
    vi = FakeLanguage("vi")
    direct = FakeTranslation("direct:")
    zh.routes["vi"] = direct
    zh.routes["en"] = FakeTranslation("pivot1:")
    en.routes["vi"] = FakeTranslation("pivot2:")

    translations, route = _resolve_route([zh, en, vi], "zh", "vi")

    assert translations == (direct,)
    assert route == ("zh", "vi")


def test_worker_uses_chinese_english_vietnamese_pivot():
    zh = FakeLanguage("zh")
    en = FakeLanguage("en")
    vi = FakeLanguage("vi")
    first = FakeTranslation("en:")
    second = FakeTranslation("vi:")
    zh.routes["en"] = first
    en.routes["vi"] = second

    translations, route = _resolve_route([zh, en, vi], "zh", "vi")

    assert translations == (first, second)
    assert route == ("zh", "en", "vi")


def test_worker_english_requires_direct_vietnamese_route():
    en = FakeLanguage("en")
    vi = FakeLanguage("vi")

    with pytest.raises(RuntimeError, match="en->vi"):
        _resolve_route([en, vi], "en", "vi")


def test_worker_reports_missing_language_package():
    en = FakeLanguage("en")

    with pytest.raises(RuntimeError, match="missing for 'vi'"):
        _resolve_route([en], "en", "vi")


def test_worker_main_emits_json_for_route(monkeypatch, capsys):
    import json
    import buzz.argos_worker as worker

    en = FakeLanguage("en")
    vi = FakeLanguage("vi")
    en.routes["vi"] = FakeTranslation("vi:")
    monkeypatch.setattr(worker, "_installed_languages", lambda: [en, vi])

    exit_code = worker.main(["route", "en", "vi"])

    assert exit_code == 0
    output = capsys.readouterr().out.strip()
    assert output.startswith("ARGOS_RESULT_JSON=")
    payload = json.loads(output.split("=", 1)[1])
    assert payload["ok"] is True
    assert payload["route"] == ["en", "vi"]


def test_worker_main_translates_stdin(monkeypatch, capsys):
    import io
    import json
    import buzz.argos_worker as worker

    en = FakeLanguage("en")
    vi = FakeLanguage("vi")
    en.routes["vi"] = FakeTranslation("vi:")
    monkeypatch.setattr(worker, "_installed_languages", lambda: [en, vi])
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO("Hello"))

    exit_code = worker.main(["translate", "en", "vi"])

    assert exit_code == 0
    output = capsys.readouterr().out.strip()
    payload = json.loads(output.split("=", 1)[1])
    assert payload["text"] == "vi:Hello"


def test_worker_main_can_write_result_file(monkeypatch, tmp_path):
    import json
    import buzz.argos_worker as worker

    en = FakeLanguage("en")
    vi = FakeLanguage("vi")
    en.routes["vi"] = FakeTranslation("vi:")
    monkeypatch.setattr(worker, "_installed_languages", lambda: [en, vi])

    result_file = tmp_path / "result.json"
    exit_code = worker.main(["route", "en", "vi", str(result_file)])

    assert exit_code == 0
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    assert payload == {"ok": True, "route": ["en", "vi"]}
