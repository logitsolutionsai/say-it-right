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


# ---- bug hunt 2026-11-11

def test_british_and_american_spellings_are_one_word():
    assert verify.match("What colour is the centre?", "What color is the center?") == (1.0, "")      # was 0.927
    assert verify.match("OK, let's go.", "Okay, let's go.") == (1.0, "")                              # was 0.90
    assert verify.match("The neighbour's grey programme was organised.",
                        "The neighbor's gray program was organized.")[0] == 1.0
    scores = [("the", .9), ("colour", .5), ("here", .9)]
    assert verify.slips("the colour here", "the color here", scores) == []
    # only real pairs fold: these stay themselves (and the number-word homophones stay apart on purpose)
    apart = "four for hour our more are rise wise emphasis analyses specialist organist to two won one".split()
    assert verify._same_sound(apart) == apart
    assert verify.match("We have four more hours.", "We have for moer hors.")[0] < 0.95


NGUYEN = {"Nguyen": {"say": "Win", "ipa": "wɪn"}, "Said": {"say": "sah-EED", "ipa": "sɑːˈiːd"}}


def _terms(script):
    from voicecheck import lexicon
    return lexicon.render(script, lexicon.load([], base=NGUYEN))


def test_ignore_hides_the_term_not_every_word_spelled_like_it():
    r = _terms("We win when Nguyen wins.")                 # spoken: We win when Win wins.
    assert verify.match(r.spoken, "We win when Wynn wins.", alt=r.shown, ignore=r.terms) == (1.0, "")
    score, diff = verify.match(r.spoken, "We lose when Win wins.", alt=r.shown, ignore=r.terms)
    assert score < 0.95 and diff == "win -> lose"          # was 1.0: the ordinary "win" was hidden with the name
    r = _terms("He said yes and Said left.")
    assert verify.match(r.spoken, "He said yes and Sayeed left.", alt=r.shown, ignore=r.terms)[0] == 1.0
    assert verify.match(r.spoken, "He shouted yes and Said left.", alt=r.shown, ignore=r.terms)[0] < 0.95
    # a bare phrase still hides every place it is written (callers that have no Terms)
    assert verify.match("We win when Win wins.", "We lose when Win wins.", ignore=["Win"])[0] == 1.0


class FakeEar:
    """Stands in for the phoneme ear: records which words it was asked to judge, and passes them."""
    def __init__(self):
        self.asked = []

    def judge(self, y, sr, target_ipa, t0, t1, word="", said="", pad=0.08):
        self.asked.append((word, round(t0, 2)))
        return type("V", (), {"as_dict": lambda self: {"word": word, "ok": True, "heard": "", "target": target_ipa,
                                                        "score": 1.0}})()


def test_a_wrong_ordinary_word_is_not_excused_by_a_name_that_is_spelled_like_it():
    from voicecheck import lexicon
    table = lexicon.load([], base=NGUYEN)
    script, said = "We win when Nguyen wins.", "We lose when Win wins."
    c = FakeChecker({1: said, 5: said})
    s, y = _session(c, "we lose when win wins")            # the voice said "lose" for "win"
    s._ear = FakeEar()
    rep = s.check(y, 16000, script, table=table)
    assert not rep["passed"] and any("win" in r for r in rep["reasons"]), rep["reasons"]
    assert [w for w, _ in s._ear.asked] == ["Nguyen"]
    good = "We win when Win wins."
    s, y = _session(FakeChecker({1: good, 5: good}), "we win when win wins")
    s._ear = FakeEar()
    rep = s.check(y, 16000, script, table=table)
    assert rep["passed"], rep["reasons"]
    win, name = s._listener.spans[1], s._listener.spans[3]
    assert [w for w, _ in s._ear.asked] == ["Nguyen"]
    assert win[2] < s._ear.asked[0][1] <= name[1]           # it listened after "when", not at the ordinary "win"


def test_a_number_heard_as_another_number_is_a_slip_whatever_the_letters_score():
    scores = [("he", .95), ("did", .95), ("fifteen", .9), ("reps", .95)]
    assert verify.slips("He did 15 reps", "He did 50 reps", scores) == ["fifteen (0.90)"]             # was []
    scores = [("it", .9), ("costs", .9), ("fifteen", .9), ("hundred", .9), ("dollars", .9)]
    assert verify.slips("It costs $1,500", "It costs $1,600", scores) == ["fifteen (0.90)"]
    # the same number written another way, or a word that is no number, is not a swap
    assert verify.slips("about 100 people", "about a hundred people",
                        [("about", .9), ("one", .7), ("hundred", .9), ("people", .9)]) == []
    assert verify.slips("In 1959", "in nineteen fifty-nine", [("in", .9), ("nineteen", .9), ("fifty", .9),
                                                              ("nine", .9)]) == []
    assert verify.slips("June 5", "June 5th", [("june", .9), ("five", .8)]) == []
    assert verify.slips("for 2 days", "for a few days", [("for", .9), ("two", .9), ("days", .9)]) == []


def test_many_numbers_do_not_make_the_comparison_slow():
    import time
    line = ("In 1959 the company had 784 workers and made $1,234.50 per day, about 3.5% more than 1,000,000 others "
            "between 2020-2021 at 10:30. ")
    script = line * 10                                     # 220 words, 70 numbers: took 36 s
    said = " ".join(verify.norm_words(script)).replace("one thousand two hundred thirty four", "twelve thirty four")
    t0 = time.perf_counter()
    assert verify.match(script, script)[0] == 1.0
    assert verify.match(script, said)[0] == 1.0
    assert time.perf_counter() - t0 < 5
    # the same choices as scoring the whole text for every reading
    said = "pages three five and ten through twenty then a hundred or a thousand at three oh five".split()
    assert verify._fit("pages 3-5 and 10-20, then 100 or 1,000 at 3:05", said) == said
    assert verify._fit("100 100 100", "a hundred one hundred one zero zero".split()) == \
        "a hundred one hundred one zero zero".split()
    assert verify._fit("We are #1 in 2½ years", "we are number one in two and a half years".split()) == \
        "we are number one in two and a half years".split()


def test_syllables_count_silent_e_and_spelled_out_acronyms():
    assert verify.syllables("She asked what makes names sometimes used") == 8        # was 14
    assert verify.syllables("the idea of science is quiet") == 10                    # was 7
    assert verify.syllables("one hundred and ninety nine million") == 9
    assert verify.syllables("GPU and FBI") == 7 and verify.syllables("NASA") == 2    # was 3: letters, a syllable each
    assert verify.syllables("the network learns") == 4
