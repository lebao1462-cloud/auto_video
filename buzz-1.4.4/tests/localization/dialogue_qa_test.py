from dataclasses import replace
import json

import pytest

from buzz.localization.dialogue_qa import auto_review_dialogue
from buzz.localization.translation import (
    TranslatedLocalizationSegment, TranslatedLocalizationTranscript,
)


def transcript(lines):
    return TranslatedLocalizationTranscript(
        source_language="zh", target_language="vi", source_file="input.mp4",
        segments=tuple(
            TranslatedLocalizationSegment(
                float(i), float(i+1), f"中文{i}", text
            ) for i, text in enumerate(lines)
        ),
    )


class Reviewer:
    translation_policy_version = "test-v5"
    model = "test-model"

    def __init__(self, transform=None):
        self.calls = []
        self.transform = transform

    def review_dialogue_alignment(self, *, segments, context, source_language, target_language):
        self.calls.append((segments, context, source_language, target_language))
        return [(x["id"], self.transform(x) if self.transform else x["current_vietnamese"])
                for x in segments]


def test_no_opt_in_review_method_keeps_transcript_unchanged(tmp_path):
    original = transcript(["Tôi thấy ổn rồi."])
    assert auto_review_dialogue(original, object(), checkpoint_path=tmp_path/"c.json") is original


def test_short_natural_lines_do_not_spend_gemini_quota(tmp_path):
    provider = Reviewer()
    original = transcript(["Anh về nhé." for _ in range(49)])
    result = auto_review_dialogue(original,provider,checkpoint_path=tmp_path/"c.json")
    assert result == original
    assert provider.calls == []


def test_long_lines_get_context_and_keep_correct_ids(tmp_path):
    source = transcript(["Xin chào." for _ in range(29)])
    lines = list(source.segments)
    lines[24] = replace(lines[24],translated_text="Anh em mình hôm nay nhất định phải đến đó ngay nhé.")
    source = replace(source,segments=tuple(lines))
    changed = "Anh em mình phải đến đó ngay."

    def transform(item):
        return changed if item["id"] == "line-000024" else item["current_vietnamese"]

    provider = Reviewer(transform)
    results = auto_review_dialogue(source,provider,checkpoint_path=tmp_path/"qa.json")
    assert len(results.segments)==29
    assert results.segments[24].translated_text==changed
    assert results.segments[24].source_text==source.segments[24].source_text
    assert results.segments[24].start==source.segments[24].start
    assert len(provider.calls)==1
    payload,context,sl,tl=provider.calls[0]
    assert sl=="zh" and tl=="vi"
    assert payload[0]["id"]=="line-000024"
    assert [c["id"] for c in context] == [
        "line-000019","line-000020","line-000021",
        "line-000022","line-000023",
    ]


def test_checkpoint_reuses_identical_input_but_changes_for_other_draft(tmp_path):
    origin = transcript(["Tôi nói cho anh biết tất cả chuyện này ngay đi."])
    first=Reviewer(lambda item:"Tôi sẽ nói cho anh biết.")
    path=tmp_path/"qa.json"
    out=auto_review_dialogue(origin,first,checkpoint_path=path)
    assert out.segments[0].translated_text=="Tôi sẽ nói cho anh biết."
    again=Reviewer(lambda item:"KHÔNG DÙNG")
    assert auto_review_dialogue(origin,again,checkpoint_path=path)==out
    assert again.calls==[]
    changed=replace(origin,segments=(replace(origin.segments[0],
                 translated_text="Tôi sẽ nói cho anh nghe toàn bộ chuyện đó nhé."),))
    auto_review_dialogue(changed,again,checkpoint_path=path)
    assert len(again.calls)==1


def test_invalid_banned_or_missing_candidates_preserve_reviewed_baseline(tmp_path):
    original=transcript(["Tôi kể cho anh nghe hết câu chuyện hôm nay này.",
                         "Anh phải nghe lời tôi nói ngay bây giờ nhé."])
    p=Reviewer(lambda x: "Mày nghe đây!" if x["id"]=="line-000000" else "")
    out=auto_review_dialogue(original,p,checkpoint_path=tmp_path/"qa.json")
    assert out==original


def test_broken_ids_bounded_two_calls_and_safe_fallback(tmp_path):
    source=transcript(["Anh phải nghe lời tôi nói ngay bây giờ nhé."])
    class Broken(Reviewer):
        def review_dialogue_alignment(self, **kwargs):
            self.calls.append(kwargs)
            return [("unknown-id","Bản sai ID")]
    p=Broken()
    assert auto_review_dialogue(source,p,checkpoint_path=tmp_path/"qa.json")==source
    assert len(p.calls)==2


def test_provider_transport_errors_surface_for_checkpoint_resume(tmp_path):
    source=transcript(["Anh phải nghe lời tôi nói ngay bây giờ nhé."])
    class Failing(Reviewer):
        def review_dialogue_alignment(self,**kwargs):
            raise RuntimeError("quota 429")
    with pytest.raises(RuntimeError, match="quota"):
        auto_review_dialogue(source,Failing(),checkpoint_path=tmp_path/"qa.json")


def test_prohibited_baseline_is_rejected_before_any_calls(tmp_path):
    provider=Reviewer()
    with pytest.raises(ValueError,match="prohibited baseline"):
        auto_review_dialogue(transcript(["Tao bảo mày rồi."]),
                             provider,checkpoint_path=tmp_path/"qa.json")
    assert not provider.calls


def test_cancellation_prevents_qa_api_call(tmp_path):
    import threading
    flag=threading.Event()
    flag.set()
    with pytest.raises(RuntimeError,match="cancelled"):
        auto_review_dialogue(transcript(["Anh phải nghe lời tôi nói ngay bây giờ nhé."]),
                             Reviewer(),cancel_event=flag,
                             checkpoint_path=tmp_path/"qa.json")


def test_proper_noun_eyebrows_not_considered_prohibited(tmp_path):
    provider=Reviewer()
    t=transcript(["Lông mày cô ấy đẹp lắm."])
    assert auto_review_dialogue(t,provider,checkpoint_path=tmp_path/"qa.json")==t


def test_workflow_runs_gemini_qa_before_any_tts(tmp_path, monkeypatch):
    from buzz.localization import workflow
    from buzz.localization.transcript import LocalizationTranscript, LocalizationSegment
    from buzz.localization.translation import (
        TranslatedLocalizationTranscript, TranslatedLocalizationSegment,
    )
    source_file=tmp_path/"sample.mp4"
    source_file.write_bytes(b"dummy")
    asr=LocalizationTranscript(source_language="zh",source_file=str(source_file),
        segments=(LocalizationSegment(0,1,"你立刻回来"),))
    raw=TranslatedLocalizationTranscript(source_language="zh",target_language="vi",
        source_file=str(source_file), segments=(
            TranslatedLocalizationSegment(0,1,"你立刻回来",
                "Anh phải về nhà ngay lập tức cho tôi."),))
    class QAProvider:
        model="fake-gemini"
        translation_policy_version="v5"
        def review_dialogue_alignment(self, **kw):
            return [(e["id"],"Anh về ngay đi.") for e in kw["segments"]]
    monkeypatch.setattr(workflow, "probe_media_duration", lambda p:10.0)
    monkeypatch.setattr(workflow, "translate_for_localization",
                        lambda *a,**k:raw)
    class StopAtTTS(Exception):
        pass
    def stop_tts(translated,*args,**kwargs):
        assert translated.segments[0].translated_text=="Anh về ngay đi."
        raise StopAtTTS()
    monkeypatch.setattr(workflow, "synthesize_for_localization", stop_tts)
    with pytest.raises(StopAtTTS):
        workflow._localize_single_video(
            str(source_file),None,"",QAProvider(),object(),
            workflow.LocalizationWorkflowOptions(output_directory=str(tmp_path)),
            transcribe_func=lambda *a,**kw:asr)
    assert (tmp_path/"sample_localization_work"/"dialogue-qa-checkpoint.json").exists()


def test_unflagged_qa_paraphrases_are_not_accepted(tmp_path):
    short="Mẹ vẫn khỏe."
    dense="Anh nhớ phải đi ngay đến chỗ tôi bảo nhé."
    initial=transcript([dense,short])
    reviewer=Reviewer(lambda item:"Mẹ cô ấy khỏe." if item["id"]=="line-000001"
                      else "Anh đến chỗ tôi ngay nhé.")
    final=auto_review_dialogue(initial,reviewer,checkpoint_path=tmp_path/"qa.json")
    assert final.segments[1].translated_text==short
    assert final.segments[0].translated_text=="Anh đến chỗ tôi ngay nhé."


def test_compression_cannot_switch_em_to_co_or_invent_pronoun(tmp_path):
    original=transcript(["Em càng nói anh càng lo, em hiểu không?"])
    p=Reviewer(lambda item:"Cô càng nói anh càng lo.")
    result=auto_review_dialogue(original,p,checkpoint_path=tmp_path/"qa.json")
    assert result==original


def test_rejects_new_typo_in_candidate(tmp_path):
    original=transcript(["Con phải nhanh chóng từ bỏ ý định đó đi nhé."])
    p=Reviewer(lambda item:"Cõ phải từ bỏ ý định đó.")
    result=auto_review_dialogue(original,p,checkpoint_path=tmp_path/"qa.json")
    assert result==original
