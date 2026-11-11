"""`vc proof`: a transcript judged against its audio. The letter model is faked (tests/conftest.py), so these run
without downloading anything; tests/test_models.py has the same checks with the real models."""
import json

import numpy as np
import pytest

from conftest import fake_listener, fake_posteriors
from voicecheck import ctc, ipa, lexicon, proof, verify

SAID = "the model is confident in wrong answers and it goes from zero to two hundred quickly"


class FakeSession:
    """A Session whose letter ear hears exactly `spoken` and whose Whisper writes `whisper`."""

    def __init__(self, spoken: str, whisper: str | None = None):
        self.listener = fake_listener(fake_posteriors(spoken))
        self.whisper = whisper if whisper is not None else spoken

    def _load(self, name, timing):
        if name == "listener":
            return self.listener
        if name == "checker":
            session = self

            class Checker:
                def transcribe(self, y, sr, beams=None):
                    return session.whisper
            return Checker()
        raise AssertionError(f"{name} should not be needed")


def run(transcript, spoken=SAID, table=None, second="whisper", whisper=None):
    doc = proof.parse(transcript) if isinstance(transcript, str) else transcript
    return doc, proof.proofread(FakeSession(spoken, whisper), None, 16000, doc, table=table or {}, second=second)


def fixed(doc, report, level="high"):
    return doc.fix(proof.edits(report, level))


# ---------------------------------------------------------------- the issues

def test_a_correct_transcript_is_clean():
    _, r = run("The model is confident in wrong answers, and it goes from zero to two hundred quickly.")
    assert r["clean"] and r["issues"] == [] and r["aligned"] and r["agreement"] == 1.0


def test_a_misheard_word_is_replaced_by_what_the_audio_says():
    doc, r = run("The model is confident and wrong answers, and it goes from zero to two hundred quickly.")
    (issue,) = r["issues"]
    assert issue["kind"] == "misheard" and issue["transcript"] == "and" and issue["suggest"] == "in"
    assert issue["confidence"] == "high" and issue["evidence"]["letters_heard"] == "in"
    assert fixed(doc, r) == "The model is confident in wrong answers, and it goes from zero to two hundred quickly."


def test_the_letter_ear_alone_still_points_at_the_misheard_word():
    # Both transcripts agree on the wrong word (a language model "corrected" both): only the letters disagree.
    wrong = "the model is confident and wrong answers and it goes from zero to two hundred quickly"
    for second, whisper in ((None, None), ("whisper", wrong)):
        _, r = run(wrong, second=second, whisper=whisper)
        (issue,) = r["issues"]
        assert issue["transcript"] == "and" and issue["evidence"]["letters_heard"] == "in"
        assert issue["kind"] in ("misheard", "unsure") and issue["confidence"] != "high"


def test_a_skipped_word_is_put_back():
    doc, r = run("It goes from zero two hundred quickly.", spoken="it goes from zero to two hundred quickly")
    (issue,) = r["issues"]
    assert issue["kind"] == "missing" and issue["suggest"] == "to" and issue["confidence"] == "high"
    assert fixed(doc, r) == "It goes from zero to two hundred quickly."


def test_a_skipped_word_is_noticed_without_a_second_transcript():
    _, r = run("It goes from zero two hundred quickly.", spoken="it goes from zero to two hundred quickly", second=None)
    (issue,) = r["issues"]                                      # spelled by ear, so never applied on its own
    assert issue["kind"] == "missing" and issue["suggest"] == "to" and issue["confidence"] != "high"
    assert issue["evidence"]["letters_heard"] == "to"


def test_a_hallucinated_sentence_is_removed_and_its_neighbour_kept():
    text = "It goes from zero to two hundred quickly. Thanks for watching."
    for second in ("whisper", None):
        doc, r = run(text, spoken="it goes from zero to two hundred quickly", second=second)
        (issue,) = r["issues"]
        assert issue["kind"] == "unheard" and issue["transcript"] == "Thanks for watching"
        assert issue["confidence"] == "high" and issue["suggest"] == ""
        assert fixed(doc, r) == "It goes from zero to two hundred quickly."


def test_numbers_written_differently_are_not_a_disagreement():
    _, r = run("In 1959 it cost $5.", spoken="in nineteen fifty nine it cost five dollars",
               whisper="In nineteen fifty-nine it cost five dollars.")
    assert r["clean"], r["issues"]


def test_a_wrong_number_is_fixed_as_a_whole():
    doc, r = run("A value such as 0.01 works.", spoken="a value such as zero point zero zero one works",
                 whisper="A value such as 0.001 works.")
    (issue,) = r["issues"]
    assert issue["kind"] == "misheard" and issue["transcript"] == "0.01" and issue["suggest"] == "0.001"
    assert fixed(doc, r, "medium") == "A value such as 0.001 works."


def test_a_name_the_two_transcripts_spell_differently_is_never_auto_fixed():
    # "Shivon" fits the letters heard better than "Siobhan" does, and it is still the wrong spelling.
    doc, r = run("Then Siobhan agreed with it.", spoken="then shivon agreed with it", whisper="Then Shivon agreed with it.")
    (issue,) = r["issues"]
    assert issue["kind"] == "spelling" and issue["suggest"] is None and issue["edit"] is None
    assert fixed(doc, r, "all") == "Then Siobhan agreed with it."


def test_a_lexicon_name_written_right_is_left_alone():
    table = {"Siobhan": lexicon.Entry("Siobhan", "Shiv-awn")}
    _, r = run("Then Siobhan agreed with it.", spoken="then shivon agreed with it", table=table,
               whisper="Then Shivon agreed with it.")
    assert r["clean"] and r["lexicon"] == [{"word": "Siobhan", "written": 1, "confirmed": 0, "doubted": 0,
                                            "restored": 0}]


def test_a_respelled_lexicon_word_is_restored_to_its_spelling():
    table = {"ReLU": lexicon.Entry("ReLU", "rel-you")}
    doc, r = run("The rel you unit is simple.", spoken="the rel you unit is simple", table=table, second=None)
    (issue,) = r["issues"]
    assert issue["kind"] == "name" and issue["transcript"] == "rel you" and issue["suggest"] == "ReLU"
    assert fixed(doc, r, "medium") == "The ReLU unit is simple."


def test_a_transcript_of_another_recording_is_called_that():
    _, r = run("Every single word of this text belongs to some other recording entirely.", second=None)
    assert not r["clean"] and not r["aligned"] and "different recording" in r["note"]
    assert all(x["confidence"] == "low" for x in r["issues"])


def test_audio_before_and_after_the_transcript_is_reported_not_smeared_over_it():
    _, r = run("It goes from zero to two hundred quickly.", second=None)
    (issue,) = r["issues"]
    assert issue["kind"] == "missing" and issue["evidence"]["letters_heard"] == "the model is confident in wrong answers and"
    assert issue["at"][0] < 0.2 and issue["confidence"] == "medium"


def test_an_empty_transcript_is_an_error():
    with pytest.raises(ValueError):
        run("[Music] (laughs)")


# ---------------------------------------------------------------- formats

SRT = """1
00:00:00,000 --> 00:00:02,500
<i>The model is confident</i>

2
00:00:02,500 --> 00:00:05,000
and wrong answers, [Music]
and it goes from zero to two hundred quickly.
"""


def test_captions_keep_their_timing_and_tags_when_fixed():
    doc = proof.parse(SRT, "srt")
    assert [doc.source[r.start:r.end] for r in doc.regions] == [
        "The model is confident", "and wrong answers, ", "and it goes from zero to two hundred quickly."]
    assert doc.regions[1].cue == 2 and doc.regions[1].t0 == 2.5
    _, r = run(doc)
    (issue,) = r["issues"]
    assert issue["suggest"] == "in" and issue["cue"] == 2 and issue["line"] == 7
    assert fixed(doc, r) == SRT.replace("and wrong", "in wrong")


def test_an_edit_across_two_captions_leaves_the_timing_line_between_them():
    doc = proof.parse(SRT, "srt")
    a = doc.source.index("confident")
    b = doc.source.index("wrong")
    out = doc.fix([(a, b, "sure of ")])
    assert "00:00:02,500 --> 00:00:05,000" in out and "<i>The model is sure of </i>" in out and "\nwrong answers" in out


def test_vtt_and_speaker_labels():
    vtt = "WEBVTT\n\nNOTE made by a tool\n\n1\n00:00.000 --> 00:02.000\n<v Maria>Hello there.\n\n00:02.000 --> 00:04.000\nSPEAKER_01: General Kenobi.\n"
    doc = proof.parse(vtt, "vtt")
    assert [doc.source[r.start:r.end].strip() for r in doc.regions] == ["Hello there.", "General Kenobi."]
    text = "Maria: Hello there.\nJon: General Kenobi.\nMaria: Note: this stays.\n(laughs) (LAUGHTER) Fine (really)."
    doc = proof.parse(text)
    assert doc.text().split("\n") == ["Hello there.", "General Kenobi.", "Note: this stays.", "Fine (really)."]
    assert proof.parse("Note: one such line is just a sentence.").text() == "Note: one such line is just a sentence."


def test_whisper_json_is_fixed_segment_by_segment():
    obj = {"text": " The model is confident and wrong answers, and it goes from zero to two hundred quickly.",
           "segments": [{"start": 0.0, "end": 2.5, "text": " The model is confident and wrong answers,",
                         "words": [{"word": "The"}]},
                        {"start": 2.5, "end": 5.0, "text": " and it goes from zero to two hundred quickly."}]}
    doc = proof.parse(json.dumps(obj), "json")
    _, r = run(doc)
    out = json.loads(fixed(doc, r))
    assert out["segments"][0]["text"] == " The model is confident in wrong answers,"
    assert out["segments"][1]["text"] == obj["segments"][1]["text"] and "words" not in out["segments"][0]
    assert out["text"] == obj["text"].replace("and wrong", "in wrong")
    with pytest.raises(ValueError):
        proof.parse('{"results": []}', "json")


def test_text_added_at_the_end_of_a_json_segment_stays_in_it():
    doc = proof.parse(json.dumps([{"text": "one two"}, {"text": "four"}]), "json")
    out = json.loads(doc.fix([(7, 7, " three")]))
    assert [s["text"] for s in out] == ["one two three", "four"]


def test_units_point_back_at_the_source_through_numbers_and_odd_characters():
    doc = proof.parse("“Dr. Nguyen’s x² cost $5 million—in 1959!”")
    us = proof.units(doc)
    shown = [(doc.source[u.start:u.end], u.part if isinstance(u.part, str) else u.part[0]) for u in us]
    assert ("Dr.", "doctor") in shown and ("$5 million", "five million dollars") in shown
    assert ("1959", "nineteen fifty nine") in shown and any(src == "Nguyen’s" for src, _ in shown)
    assert all(doc.source[u.start:u.end].strip() for u in us)


def test_read_takes_a_file_or_the_text(tmp_path):
    f = tmp_path / "t.srt"
    f.write_text("﻿" + SRT, encoding="utf-8")
    assert proof.read(str(f)).kind == "srt" and len(proof.read(str(f)).regions) == 3
    assert proof.read("Just the words.").kind == "text"
    with pytest.raises(FileNotFoundError):
        proof.read("missing.srt")


# ---------------------------------------------------------------- the pieces underneath

def test_spot_finds_every_place_a_name_is_said_and_nothing_else():
    heard = "ð ə m ɑ d ə l ʃ ɪ v ɔ n ə ɡ ɹ iː d æ n d ʃ ə v ɑ n l ɛ f t".split()
    hits = ipa.spot("ʃɪˈvɔːn", heard)
    assert sorted((h["start"], h["end"]) for h in hits) == [(7, 12), (20, 25)]
    assert hits[0]["score"] == 1.0 and hits[0]["hard"] == 0 and hits[0]["heard"] == "ʃ ɪ v ɔ n"
    assert ipa.spot("ˈsɛɡɛdi", heard) == [] and ipa.spot("ʃɪˈvɔːn", []) == []
    one = ipa.compare("ʃɪˈvɔːn", heard[5:14])               # the same distance as compare()
    assert one.score == hits[0]["score"] and one.diff == hits[0]["diff"]


def test_greedy_and_hear_read_the_posteriors_without_a_script():
    lp = fake_posteriors("zero to two")
    lis = fake_listener(lp)
    words = lis.hear(lp)
    assert [w for w, *_ in words] == ["zero", "to", "two"]
    assert all(b > a and p > 0.9 for _, a, b, p, _ in words) and words[0][4] is None and words[1][4] is not None
    assert ctc.greedy(lp[:0]) == []


def test_a_long_take_aligned_in_sections_scores_like_one_aligned_whole(monkeypatch):
    text = " ".join([SAID] * 6)
    lp = fake_posteriors(text)
    lis = fake_listener(lp)
    units = text.split()
    whole = lis.score_units(lp, units)
    monkeypatch.setattr(verify, "LONG_FRAMES", 100)
    cut = []
    real = ctc.sections
    monkeypatch.setattr(ctc, "sections", lambda *a, **k: cut.append(real(*a, **{**k, "every": 150})) or cut[-1])
    parts = lis.score_units(lp, units)
    assert len(cut[0]) > 3                                      # it really was cut
    assert parts["words"] == whole["words"] and parts["frames"] == whole["frames"]
    assert np.allclose(parts["scores"], whole["scores"]) and parts["measured"]


def test_a_stretch_with_nothing_in_common_is_left_unmeasured_not_built(monkeypatch):
    lp = fake_posteriors(SAID)
    lis = fake_listener(lp)
    monkeypatch.setattr(verify, "MAX_CELLS", 1000)
    r = lis.score_units(lp, SAID.split())
    assert not r["measured"] and "too little in common" in r["note"] and set(r["scores"]) == {0.0}


def test_a_words_place_ignores_a_stray_letter():
    assert verify._body([10, 11, 12, 13, 90]) == (10, 14) and verify._body([5]) == (5, 6) and verify._body([]) is None
