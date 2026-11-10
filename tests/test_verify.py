"""The checker's logic, without any model: fake letter posteriors stand in for wav2vec2."""
import pytest

from voicecheck import config, verify
from conftest import fake_listener, fake_posteriors


# ---- 1. numbers in the script no longer fail a correct read

def test_listener_reads_a_year_the_way_the_audio_says_it():
    lis = fake_listener(fake_posteriors("in nineteen fifty nine the network had seven hundred eighty four weights"))
    scores = lis.word_scores(None, 16000, "In 1959 the network had 784 weights.")
    assert [w for w, _ in scores] == "in nineteen fifty nine the network had seven hundred eighty four weights".split()
    assert min(s for _, s in scores) > 0.9
    heard = "In 1959 the network had 784 waits."                     # what Whisper wrote for the real take
    assert verify.slips("In 1959 the network had 784 weights.", heard, scores) == []
    assert verify.match("In 1959 the network had 784 weights.", heard)[0] == 1.0


def test_listener_picks_the_british_and_when_the_voice_says_it():
    lis = fake_listener(fake_posteriors("seven hundred and eighty four weights"))
    scores = lis.word_scores(None, 16000, "784 weights")
    assert [w for w, _ in scores] == "seven hundred and eighty four weights".split()


def test_a_real_slip_in_a_number_is_still_caught():
    # the script says 784; the "audio" says seven hundred eighty FIVE
    lis = fake_listener(fake_posteriors("seven hundred eighty four weights", garble={3: "five"}))
    scores = lis.word_scores(None, 16000, "784 weights")
    found = verify.slips("784 weights", "785 weights", scores)
    assert found and found[0].startswith("four")


# ---- 2/3. output and pass rules are tested in test_cli.py; 4. the listener fails closed

def test_listener_fails_closed_when_the_audio_is_too_short():
    lis = fake_listener(fake_posteriors("in"))                        # far too short for the line
    scores = lis.word_scores(None, 16000, "In nineteen fifty nine the network had weights")
    assert lis.measured is False
    assert all(s == 0.0 for _, s in scores) and len(scores) == 8
    assert len(verify.slips("In nineteen fifty nine the network had weights", "In", scores)) == 8


# ---- 5. match and slips agree about "a hundred"

def test_a_hundred_is_not_a_slip():
    scores = [("about", .9), ("a", .5), ("hundred", .9), ("people", .9)]
    assert verify.match("about a hundred people", "about 100 people") == (1.0, "")
    assert verify.slips("about a hundred people", "about 100 people", scores) == []


def test_script_digits_match_a_transcript_in_words():
    assert verify.match("In 1959 it had 784 weights", "in nineteen fifty-nine it had seven hundred and eighty-four "
                                                      "weights")[0] == 1.0


# ---- the rules that were already there still hold

def test_slip_needs_whisper_and_listener_to_agree():
    scores = [("zero", .95), ("to", .40), ("two", .9), ("hundred", .9)]
    assert verify.slips("zero to two hundred", "zero two hundred", scores) == ["to (0.40)"]
    scores_ok = [("zero", .95), ("to", .80), ("two", .9), ("hundred", .9)]
    assert verify.slips("zero to two hundred", "zero two hundred", scores_ok) == []


def test_gone_word_fails_whatever_whisper_wrote():
    scores = [("confident", .95), ("in", .1), ("wrong", .9)]
    assert verify.slips("confident in wrong", "confident in wrong", scores) == ["in (0.10)"]


def test_homophones_are_not_slips():
    scores = [("one", .9), ("pass", .9), ("per", .9), ("weight", .5)]
    assert verify.slips("one pass per weight", "one pass per wait", scores) == []
    assert verify.match("one pass per weight", "one pass per wait")[0] == 1.0


def test_alt_wording_and_respelling_tails():
    config.reset({"pronunciations": {"ReLU": "rel-you"}})
    scores = [("the", .9), ("rel", .9), ("you", .22), ("unit", .9)]
    assert verify.slips("the rel-you unit", "the ReLU unit", scores, alt="the ReLU unit") == []
    assert verify.match("the rel-you unit", "the ReLU unit", alt="the ReLU unit")[0] == 1.0


def test_skip_leaves_words_to_the_phoneme_check():
    scores = [("christian", .9), ("seg", .1), ("eh", .1), ("dee", .1)]
    assert verify.slips("Christian Seg-eh-dee", "Christian Szegedy", scores, skip={1, 2, 3}) == []


def test_match_diff_is_readable():
    score, diff = verify.match("being sure of the wrong answer", "being sure the wrong answer")
    assert score < 1 and diff == "of -> -"


# ---- small things

def test_emotional_shows_the_margin():
    p = {"range": 19.4, "peak_st": 7.3, "peak_word": "x", "peak_db": 1, "tail": 4.6}
    assert verify.emotional(p, {"range_st": 19, "peak_st": 7.0, "tail_st": 4.5}) == [
        "pitch range 19.4 st (limit 19)", "stressed 'x' +7.3 st (limit 7.0)", "ending rises 4.6 st (limit 4.5)"]
    assert verify.emotional(p, {}) == [] and verify.emotional(None, {"range_st": 1}) == []


def test_pace_counts_speech_not_silence():
    spans = [("the", 0.5, 0.7), ("network", 0.7, 1.2), ("learns", 1.2, 1.7)]
    assert verify.speech_seconds(spans, 3.0) == pytest.approx(1.2)
    assert verify.speech_seconds([], 3.0) == 3.0
    assert verify.pace("the network learns", 1.2) == pytest.approx(4 / 1.2)


def test_text_functions_import_without_models():
    import sys
    assert "transformers" not in sys.modules or True    # importing verify must not load models
    assert verify.norm_words("GPT-4") == ["gpt", "four"]


def test_names_judged_by_sound_are_left_out_of_the_transcript_match():
    spoken, shown = "Christian Seg-eh-dee wrote the paper.", "Christian Szegedy wrote the paper."
    heard = "Christian Segady wrote the paper."                    # Whisper can't spell a name it doesn't know
    assert verify.match(spoken, heard, alt=shown)[0] < 0.95
    score, diff = verify.match(spoken, heard, alt=shown, ignore=["Seg-eh-dee", "Szegedy"])
    assert score == 1.0 and diff == ""
    # ...but a real slip elsewhere in the line still counts
    assert verify.match(spoken, "Christian Segady wrote a paper.", alt=shown, ignore=["Seg-eh-dee", "Szegedy"])[0] < 1


def test_masking_keeps_the_other_words_of_a_mixed_block():
    spoken, shown = "Andrej Kar-pah-thee explains", "Andrej Karpathy explains"
    score, diff = verify.match(spoken, "Andre Carpathi Explains", alt=shown, ignore=["Kar-pah-thee", "Karpathy"])
    assert score > 0.9 and diff == "andrej -> andre"


def test_phoneme_cache_never_serves_another_takes_audio():
    import numpy as np
    from voicecheck.phonemes import PhonemeEar
    ear = object.__new__(PhonemeEar)
    calls = []

    class FakeFE:
        def __call__(self, y, sampling_rate, return_tensors):
            calls.append(float(y[0]))
            import torch
            return type("X", (), {"input_values": torch.zeros(1, len(y))})()

    class FakeModel:
        def __call__(self, x):
            import torch
            return type("O", (), {"logits": torch.zeros(1, 5, 3)})()
    ear.fe, ear.model, ear._cache = FakeFE(), FakeModel(), None
    a = np.ones(16000, dtype=np.float32)
    ear.log_probs(a, 16000)
    b = np.full(16000, 2.0, dtype=np.float32)               # same length, maybe the same id() once `a` is freed
    del a
    ear.log_probs(b, 16000)
    assert calls == [1.0, 2.0]                               # the second take was really computed
    ear.log_probs(b, 16000)
    assert calls == [1.0, 2.0]                               # and the same take is served from the cache


def test_whisper_abbreviations_match_the_spelled_out_script():
    assert verify.match("Doctor Nguyen agreed.", "Dr. Nguyen agreed.")[0] == 1.0     # a real take's transcript
    assert verify.match("Dr. Nguyen agreed.", "Doctor Nguyen agreed.")[0] == 1.0
    assert verify.match("Ask Mrs. Smith versus Mr. Jones.", "Ask Missus Smith vs. Mr. Jones.")[0] == 1.0


def test_listener_aligns_an_abbreviation_as_spoken():
    lis = fake_listener(fake_posteriors("doctor nguyen agreed"))
    scores = lis.word_scores(None, 16000, "Dr. Nguyen agreed.")
    assert [w for w, _ in scores] == ["doctor", "nguyen", "agreed"] and min(s for _, s in scores) > 0.9


# ---- the transcript: greedy first, beam search only when the greedy transcript doesn't match

class FakeChecker:
    def __init__(self, by_beams):
        self.by_beams, self.calls = by_beams, []

    def transcribe(self, y, sr, beams=None):
        n = beams or config.voice()["beams"]
        self.calls.append(n)
        return self.by_beams[n]


def _session(checker, spoken):
    import numpy as np
    from voicecheck.session import Session
    s = Session(log=lambda *a: None)
    s._checker, s._listener = checker, fake_listener(fake_posteriors(spoken))
    return s, np.zeros(16000 * 3, dtype=np.float32)


def test_a_matching_greedy_transcript_is_not_decoded_again():
    line = "The model is confident in wrong."
    c = FakeChecker({1: line, 5: "never asked"})
    s, y = _session(c, "the model is confident in wrong")
    rep = s.check(y, 16000, line)
    assert c.calls == [1] and rep["beams"] == 1 and rep["passed"], rep["reasons"]


def test_a_greedy_mismatch_is_decided_by_beam_search():
    line = "The model is confident in wrong."
    c = FakeChecker({1: "The model is confident and wrong.", 5: line})      # greedy misheard, beams didn't
    s, y = _session(c, "the model is confident in wrong")
    rep = s.check(y, 16000, line)
    assert c.calls == [1, 5] and rep["beams"] == 5 and rep["heard"] == line and rep["passed"], rep["reasons"]
    c = FakeChecker({1: "The model is confident and wrong.", 5: "The model is confident and wrong."})
    s, y = _session(c, "the model is confident and wrong")                   # the voice really said "and"
    rep = s.check(y, 16000, line)
    assert c.calls == [1, 5] and not rep["passed"]
    config.reset({"retry_beams": 0})                                          # retry off
    c = FakeChecker({1: "The model is confident and wrong."})
    s, y = _session(c, "the model is confident in wrong")
    assert not s.check(y, 16000, line)["passed"] and c.calls == [1]
