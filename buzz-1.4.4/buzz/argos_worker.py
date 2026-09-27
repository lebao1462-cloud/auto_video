import json
import sys
from pathlib import Path


def _installed_languages():
    from argostranslate import translate as argos_translate

    return argos_translate.get_installed_languages()


def _find_language(languages, code):
    return next((language for language in languages if language.code == code), None)


def _resolve_route(languages, source_language, target_language="vi"):
    if source_language not in {"en", "zh"}:
        raise ValueError("Source language must be 'en' or 'zh'")
    if target_language != "vi":
        raise ValueError("Argos localization target must be 'vi'")

    source = _find_language(languages, source_language)
    target = _find_language(languages, target_language)
    if source is None or target is None:
        missing = source_language if source is None else target_language
        raise RuntimeError(f"Argos language package is missing for '{missing}'.")

    direct = source.get_translation(target)
    # Argos may synthesize a CompositeTranslation automatically. Treat only a
    # non-composite translation as a true direct package so route reporting is
    # accurate and the explicit Chinese pivot remains deterministic.
    if direct is not None and direct.__class__.__name__ != "CompositeTranslation":
        return (direct,), (source_language, target_language)

    if source_language == "zh":
        english = _find_language(languages, "en")
        if english is None:
            raise RuntimeError(
                "Argos Chinese translation requires zh->vi directly or the "
                "zh->en + en->vi pivot route; English package is missing."
            )
        first = source.get_translation(english)
        second = english.get_translation(target)
        if first is not None and second is not None:
            return (first, second), ("zh", "en", "vi")

        if direct is not None:
            return (direct,), ("zh", "en", "vi")

    raise RuntimeError(
        f"Argos translation route '{source_language}->vi' is not installed."
    )


def _emit(payload, result_file=None):
    serialized = json.dumps(payload, ensure_ascii=True)
    if result_file:
        Path(result_file).write_text(serialized, encoding="utf-8")
    else:
        print("ARGOS_RESULT_JSON=" + serialized, flush=True)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        _emit({"ok": False, "error": "Argos worker command is incomplete."})
        return 2

    command = args[0]
    source_language = args[1]
    target_language = args[2] if len(args) > 2 else "vi"
    result_file = args[3] if len(args) > 3 else None

    try:
        translations, route = _resolve_route(
            _installed_languages(), source_language, target_language
        )
        if command == "route":
            _emit({"ok": True, "route": list(route)}, result_file)
            return 0
        if command != "translate":
            raise ValueError(f"Unknown Argos worker command: {command}")

        text = sys.stdin.read()
        if not text.strip():
            raise ValueError("Translation text cannot be empty")

        translated = text.strip()
        for translation in translations:
            translated = translation.translate(translated)
        if not isinstance(translated, str) or not translated.strip():
            raise RuntimeError("Argos returned empty translation text")
        _emit(
            {
                "ok": True,
                "route": list(route),
                "text": translated.strip(),
            },
            result_file,
        )
        return 0
    except Exception as exc:
        _emit({"ok": False, "error": str(exc)}, result_file)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
