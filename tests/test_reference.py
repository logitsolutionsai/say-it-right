"""The bugs fixed in reference/ (OpenMontage audits copied here), tested through their pure functions."""
from conftest import stub_openmontage, unstub_pil

stub_openmontage()

import lecture_voice_listen  # noqa: E402
import number_audit  # noqa: E402
import small_words  # noqa: E402

unstub_pil()                     # the fake Pillow was only for number_audit's import; other tests load transformers


def test_voice_listen_audits_the_name_not_the_pronoun():
    terms = ["He", "Ba", "eta", "Adam"]
    exact, lower = set(terms), {t.lower(): t for t in terms}
    hit = {w: lecture_voice_listen.is_term(w, exact, lower) for w in ["He", "he", "Ba", "ba", "Adam", "adam"]}
    assert hit == {"He": True, "he": False, "Ba": True, "ba": False, "Adam": True, "adam": True}


def test_small_words_catches_a_dropped_word_between_long_neighbours():
    assert not small_words.heard_ok("to", ["parameters"], ["gradients"], "parameters gradients")
    assert not small_words.heard_ok("to", ["zero"], ["two"], "zero two")
    assert small_words.heard_ok("to", ["parameters"], ["gradients"], "parameters to gradients")
    assert small_words.heard_ok("to", ["parameters"], ["gradients"], "parameterstogradients")


def test_number_audit_reads_hundreds_after_a_teen():
    v = number_audit.value_of
    assert v("twelve hundred fifty".split()) == 1250
    assert v("thirteen hundred and twenty".split()) == 1320
    assert v("fifteen hundred".split()) == 1500
    assert v("nineteen fifty nine".split()) == 1959
    assert v("two hundred and twenty".split()) == 220
