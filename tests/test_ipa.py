import pytest

from voicecheck import ipa
from voicecheck.phonemes import hard_errors


@pytest.mark.parametrize("ipa_str, name, want", [
    ("ˈsɛɡɛdi", True, "Seg-eh-dee"),           # matches the respelling the OpenMontage lexicon arrived at by ear
    ("ʃɪˈvɔːn", True, "Shiv-awn"),             # Wikipedia: shiv-AWN
    ("ˈsɜːrʃə", True, "Sur-shuh"),
    ("wɑːˈkiːn", True, "Wah-keen"),
    ("sɛrˈɡeɪ", True, "Sair-gay"),
    ("ˈdʒɛfri ˈhɪntən", True, "Jeff-ree Hin-tun"),     # "Jef" doubled like "of" -> "off"
    ("pəˈtɑːtoʊ", False, "puh-tah-toh"),       # p/t/k before a stressed vowel starts the syllable
    ("ˈɑːndrɛj", True, "Ahn-dray"),
    ("ˈɡiːzə", True, "Ghee-zuh"),              # g before e/i stays hard
    ("ʒɑ̃", True, "Zhahn"),
])
def test_respell(ipa_str, name, want):
    assert ipa.respell(ipa_str, name=name) == want


def test_stress_style_capitalises_only_multisyllable_stress():
    assert ipa.respell("ʃɪˈvɔːn", style="stress") == "shiv-AWN"
    assert ipa.respell("ˈwɪn", style="stress") == "win"            # never a lone all-caps word
    assert ipa.candidates("ˈsɛɡɛdi", name=True) == ["Seg-eh-dee", "SEG-eh-dee", "Seg eh dee"]


def test_compare_scores_the_target_not_the_neighbours():
    assert ipa.compare("ˈsɛɡɛdi", "s ɛ ɡ ɛ d i").score == 1.0
    assert ipa.compare("ˈsɛɡɛdi", "k ɹ ɪ s tʃ ə n s ɛ ɡ ɛ d i w ɹ oʊ t").score == 1.0


def test_compare_orders_pronunciations_sensibly():
    t = "ˈsɛɡɛdi"
    close = ipa.compare(t, "s e ɡ ə d i").score              # vowel colour only
    soft_g = ipa.compare(t, "s ɛ dʒ ɛ d i")                    # a real mispronunciation
    other = ipa.compare(t, "k ɹ ɪ s tʃ ə n").score            # another word
    assert close > soft_g.score > other
    assert hard_errors(soft_g.diff) == 1 and hard_errors(ipa.compare(t, "s e ɡ ə d i").diff) == 0


def test_units_split_diphthongs_and_drop_length():
    assert [u[0] for u in ipa.units("ˈɡeɪt")] == ["g", "e", "ɪ", "t"]
    assert [u[0] for u in ipa.units("iː")] == ["i"]
    assert [u[0] for u in ipa.units("ˈsɜːrʃə")] == ["s", "ɜ", "R", "ʃ", "ə"]          # r after a vowel is weak


def test_syllables_that_spell_real_words_are_doubled():
    assert ipa.respell("ɪˈɒfɛ", name=True) == "Ih-off-eh"          # not "Ih-of-eh", read "uv"
    assert ipa.respell("ˈɪsə") == "iss-uh"                         # not "is-uh", read "iz"
    assert ipa.respell("ˈɪzə") == "iz-uh"                          # a real z stays single
