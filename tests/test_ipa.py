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


# ---- IPA as the sources really write it (bug hunt 2026-11-11)

def test_tie_bars_are_one_sound():
    assert ipa.respell("t͡ʃɜːrt͡ʃ") == "church"                    # was "tshurtsh"
    assert ipa.respell("d͜ʒʌd͜ʒ") == "juj"
    c = ipa.compare("t͡ʃɪp", ["tʃ", "ɪ", "p"])                     # was 0.629 with a missing t: a right take failed
    assert c.score == 1.0 and hard_errors(c.diff) == 0
    assert ipa.respell("ˈʧɪp") == "chip" and ipa.compare("ʧɪp", ["tʃ", "ɪ", "p"]).score == 1.0     # the old ligature


def test_the_first_of_several_variants_is_the_target():
    assert ipa.respell("ˈniːtʃə, -tʃi", name=True) == "Nee-chuh"   # was "Nee-chuh Chee"
    assert ipa.respell("ˈiːðər; ˈaɪðər") == "ee-dher" and ipa.respell("ˈiːðər or ˈaɪðər") == "ee-dher"
    c = ipa.compare("/ˈniːtʃə, -tʃi/", "ˈniːtʃə")                  # was 0.647 with two sounds "missing"
    assert c.score == 1.0 and c.target == "n i tʃ ə"
    assert [u[0] for u in ipa.units("ˌɪntərˈnæʃənəl")][:3] == ["ɪ", "n", "t"]       # ˌ itself is no variant mark
    assert [u[0] for u in ipa.units("'sɛ,ɡɛdi")] == list("sɛgɛdi")                  # a comma inside a word: stress


@pytest.mark.parametrize("ipa_str, want", [
    ("ŋwiən˧˥", "ngwee-un"),                   # tone letters
    ("ɕi˥˩ tɕin˥˩ pʰiŋ˧˥", "shee cheen peeng"),
    ("kʼaːtʼ", "kaht"),                        # ejectives
    ("bɜ˞d", "burd"),                          # r-colouring written as a hook
    ("ˈbʌtn̩", "but-un"),                       # a syllabic consonant needs a vowel letter: was "butn"
    ("ˈlɪtl̩", "lit-ul"),
    ("ˈbʌtᵊn", "but-un"),
    ("ˈfʁiːdʁɪç", "freed-rikh"),               # ç is not c + nothing: was "freed-riky"
    ("dɔʏ̯tʃ", "doych"),                        # a non-syllabic vowel is half a diphthong: was "daw-uuch"
    ("mɑʊ̯", "mow"),
    ("xu̯an", "khwahn"),
    ("taɪm", "tyme"),                          # "tym" reads "tim"
    ("naɪs", "nyss"),
    ("ˈaɪnstaɪn", "eyen-styne"),               # Wikipedia's EYEN-styne: was "yn-styn"
    ("ˈaɪlə", "eye-luh"),
    ("skaɪ", "sky"),
    ("ɓa ɗa ǃa", "bah dah kah"),                # sounds with no English letter: the nearest one
    ("hello!", "hel-loh"),
    ("ʼ", ""),
])
def test_respell_gives_letters_only(ipa_str, want):
    assert ipa.respell(ipa_str) == want
    for text in [ipa.respell(ipa_str, "stress", True)] + ipa.candidates(ipa_str, name=True):
        assert all(c.isascii() and (c.isalpha() or c in "- ") for c in text), text


def test_compare_is_unchanged_for_plain_ipa():
    assert ipa.compare("ʃɪˈvɔːn", ["ʃ", "ɪ", "v", "ɔ", "n"]).score == 1.0
    assert ipa.compare("ˈsɛɡɛdi", "ˈsɛɡɛdi").score == 1.0
    assert "!" in ipa.compare("ʃɪˈvɔːn", ["j", "ɑ", "b", "ə", "n"]).diff
    # the marks that only respelling reads leave the comparison as it was
    assert [u[0] for u in ipa.units("ˈbʌtn̩")] == ["b", "ʌ", "t", "n"]
    assert [u[0] for u in ipa.units("mɑʊ̯")] == ["m", "ɑ", "ʊ"]
    assert [u[0] for u in ipa.units("ɪç")] == ["ɪ", "c"] and ipa.compare("ɪç", ["ɪ", "ç"]).score == 1.0
    assert [u[0] for u in ipa.units("ŋwiən˧˥")] == ["ŋ", "w", "i", "ə", "n"]        # a tone is no sound to miss


def test_an_empty_target_matches_nothing():
    assert ipa.compare("", "k æ t").score == 0.0 and ipa.compare("/ /", ["k"]).score == 0.0
