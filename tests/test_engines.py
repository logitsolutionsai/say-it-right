from voicecheck import engines
from voicecheck.lexicon import Entry
from voicecheck.scan import scan

TABLE = {"Szegedy": Entry("Szegedy", "Seg-eh-dee", ipa="ˈsɛɡɛdi"), "ReLU": Entry("ReLU", "rel-you")}
LINE = "Szegedy used ReLU."


def test_text_and_ssml():
    assert engines.to_engine(LINE, TABLE, "text") == "Seg-eh-dee used rel-you."
    assert engines.to_engine(LINE, TABLE, "ssml") == (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        '<phoneme alphabet="ipa" ph="ˈsɛ.gɛ.di">Szegedy</phoneme> used <sub alias="rel-you">ReLU</sub>.</speak>')
    azure = engines.ssml(LINE, TABLE, voice="en-US-AvaNeural")
    assert '<voice name="en-US-AvaNeural">' in azure and azure.endswith("</voice></speak>")
    assert engines.ssml(LINE, TABLE, bare=True).startswith("<speak><phoneme")


def test_possessives_carry_their_s_into_the_ipa():
    table = {"Nguyen": Entry("Nguyen", "Win", ipa="wɪn"), "Ruiz": Entry("Ruiz", "Roo-eess", ipa="ruˈiːs"),
             "Kraft": Entry("Kraft", "Kraft", ipa="kræft")}
    out = engines.ssml("Nguyen's, Ruiz's and Kraft's teams", table, bare=True)
    assert 'ph="wɪnz">Nguyen&#x27;s<' in out and 'ph="ɹu.ˈi.sɪz"' in out and 'ph="kɹæfts"' in out


def _ph(ipa, shown="Word", **kw):
    """The ph="" values the ssml format gives a word with this IPA."""
    import re
    out = engines.ssml(f"{shown}.", {shown: Entry(shown, "x", ipa=ipa)}, bare=True, **kw)
    return re.findall(r'ph="([^"]*)"', out)


def test_ssml_marks_every_syllable_so_azure_stresses_the_right_one():
    # Azure: "to stress one syllable by placing stress symbol before this syllable, you need to mark all
    # syllables for the word. Or else, the syllable before this stress symbol is stressed."
    assert _ph("ʃɪˈvɔːn") == ["ʃɪ.ˈvɔn"]
    assert _ph("ˌkuːbərˈnɛtiːz") == ["ˌku.bɚ.ˈnɛ.tiz"]
    assert _ph("əˈmɛrɪkə") == ["ə.ˈmɛ.ɹɪ.kə"]
    assert _ph("wɪn") == ["wɪn"]


def test_ssml_has_only_phones_azure_en_us_accepts():
    import pytest
    assert _ph("tɕinpʰiŋ") == ["tʃin.piŋ"]                     # no aspiration mark
    assert _ph("t͡ʃaɪˈkɒf.ski") == ["tʃaɪ.ˈkɑf.ski"]            # no tie bar
    assert _ph("ˈbʌtn̩") == ["ˈbʌ.tən"]                         # a syllabic consonant is schwa + consonant
    assert _ph("ˈmʏnçn̩") == ["ˈmʊn.hən"]
    assert _ph("bax") == ["bɑk"]                               # a bare a is not an en-US phone
    assert _ph("ʈʂɑŋ") == ["tʃɑŋ"] and _ph("ˈbõ") == ["ˈboʊn"]
    with pytest.raises(ValueError, match=r"Xhosa: 'ǁ' \(U\+01C1\)"):   # a click: no English sound, so no document
        _ph("ǁʰɔːsa", shown="Xhosa")
    assert _ph("ˈmʏnçn̩", lang="de-DE") == ["ˈmʏnçn̩"]          # another language's voice gets the IPA as written


def test_ssml_gives_each_word_of_a_phrase_its_own_phoneme():
    import pytest
    table = {"Xi Jinping": Entry("Xi Jinping", "shee jin-ping", ipa="ɕi tɕinpʰiŋ")}
    out = engines.ssml("Xi  Jinping's visit", table, bare=True)
    assert out == ('<speak><phoneme alphabet="ipa" ph="ʃi">Xi</phoneme>  '
                   '<phoneme alphabet="ipa" ph="tʃin.piŋz">Jinping&#x27;s</phoneme> visit</speak>')
    assert _ph("ʒɑ̃ lyk", shown="Jean-Luc") == ["ʒɑn.luk"]      # one written word, however many in the IPA
    with pytest.raises(ValueError, match="New York: the IPA has 1 words for 2 written"):
        engines.ssml("New York", {"New York": Entry("New York", "noo york", ipa="nuːˈjɔːrk")})


def test_us_english_phones_for_azure_and_cartesia():
    assert engines.us_english("ˈsɜːrʃə") == "ˈsɝʃə"            # r-coloured, no length mark
    assert engines.us_english("ˈjofːɛ") == "ˈjoʊfɛ"            # a lone o is the English oʊ
    assert engines.us_english("ˈɑːndrɛj") == "ˈɑndɹeɪ"          # ɛj is the English eɪ, r is ɹ
    assert engines.us_english("ˈbaχ") == "ˈbɑk"                # neither χ nor a bare a is an English phone
    assert engines.us_english("ˈwɔːtə(r)") == "ˈwɔtɚ"          # the optional (r) is said in the US; no parentheses
    assert engines.us_english("boɪ") == "bɔɪ"                  # not boʊɪ
    assert engines.us_english("ã") == engines.us_english("ã") == "ɑn"   # ã, composed or not


def test_no_format_leaks_a_phone_its_engine_lacks():
    assert engines.misaki("ˈnɪər") == "nˈɪɹ"                   # not ɪɹɹ
    assert engines.misaki("ʒɑ̃") == "ʒɑn" and engines._espeak("ʒɑ̃") == "ʒɑːn"   # the nasal vowel keeps its n
    assert engines.misaki("bɔ̃ʒuʁ") == "bɔnʒuɹ"                # no ʁ
    assert engines.us_english("aɪˈdɪə") == "aɪˈdɪə"            # British ɪə: no r is invented ("idear")


def test_terms_are_replaced_where_they_stand():
    from types import SimpleNamespace
    from voicecheck.lexicon import Rendered, Term
    ai = {"AI": Entry("AI", "A I", ipa="ˌeɪˈaɪ")}
    assert engines.to_engine("The MAIN idea of AI.", ai, "ssml", {"bare": True}) == (
        '<speak>The MAIN idea of <phoneme alphabet="ipa" ph="ˌeɪ.ˈaɪ">AI</phoneme>.</speak>')
    relu = {"ReLU": Entry("ReLU", "rel-you", ipa="ˈrɛljuː")}
    assert engines.to_engine("Unrelated ReLUs, then ReLU.", relu, "elevenlabs") == "Unrelated ReLUs, then /ˈrɛljuː/."
    if "at" in Term.__dataclass_fields__:                      # the lexicon records each term's offset
        assert engines.to_engine("a cat {a|ay} here", {}, "elevenlabs") == "a cat ay here"
    # a term that says where it stands is trusted over a search (the first "a" here is an ordinary word)
    term = SimpleNamespace(shown="a", said="ay", ipa=None, key=None, occurrence=0, at=6)
    r = Rendered(spoken="a cat ay here", shown="a cat a here", ssml="", terms=[term])
    assert engines._with(r, lambda t: t.said) == "a cat ay here"
    term.at = -1                                               # unknown: the old search, whole words first
    assert engines._with(r, lambda t: t.said) == "ay cat a here"


def test_control_characters_never_reach_the_xml():
    import xml.etree.ElementTree as ET
    out = engines.ssml("soft\x0bbreak \x01 Szegedy\x0cReLU￾", TABLE, bare=True)
    assert ET.fromstring(out).tag == "speak" and out.startswith("<speak>soft break   <phoneme")
    phrase = {"Xi\x0bJinping": Entry("Xi\x0bJinping", "shee jin-ping", ipa="ɕi tɕinpʰiŋ")}
    assert ET.fromstring(engines.ssml("Xi\x0bJinping", phrase, bare=True)).tag == "speak"
    x = engines.pls({"bell": Entry("bell", "ding\x07dong"), "b\x1fell": Entry("b\x1fell", "x", ipa="bɛl")})
    assert ET.fromstring(x.encode("utf-8")).tag.endswith("lexicon") and "<alias>ding dong</alias>" in x


def test_inline_ipa_engines():
    assert engines.to_engine(LINE, TABLE, "elevenlabs") == "/ˈsɛɡɛdi/ used rel-you."
    assert engines.to_engine(LINE, TABLE, "cartesia") == "<<ˈ|s|ɛ|ɡ|ɛ|d|i>> used rel-you."
    assert engines.to_engine(LINE, TABLE, "kokoro") == "[Szegedy](/sˈɛɡɛdi/) used rel-you."
    assert engines.to_engine(LINE, TABLE, "piper") == "[[ sˈɛɡɛdi ]] used rel-you."


def test_cartesia_adds_the_required_stress():
    assert engines.cartesia("wɪn") == "<<ˈ|w|ɪ|n>>"


def test_cartesia_uses_only_sonic_3_symbols():
    import pytest
    assert engines.cartesia("ləˈkʌn") == "<<l|ə|ˈ|k|ə|n>>"                 # no ʌ: the cup vowel is ə
    assert engines.cartesia("ˈwɔːtər") == "<<ˈ|w|ɔ|t|ɝ>>"                  # no ɚ: ɝ is the only r-coloured vowel
    assert engines.cartesia("ɡəˈrɑːʒ") == "<<ɡ|ɝ|ˈ|ɑ|ʒ>>"                  # never ə|ɹ, even across a stress mark
    assert engines.cartesia("əˈmɛrɪkə") == "<<ə|ˈ|m|ɛ|ɹ|ɪ|k|ə>>"           # ɹ stays after a full vowel
    assert engines.cartesia("t͡ʃaɪˈkɒf.ski") == "<<tʃ|aɪ|ˈ|k|ɑ|f|s|k|i>>"   # an affricate is one symbol; no dots
    assert engines.cartesia("bax") == "<<ˈ|b|ɑ|k>>"
    assert engines.cartesia("ɕi tɕinpʰiŋ") == "<<ˈ|ʃ|i>> <<ˈ|tʃ|i|n|p|i|ŋ>>"
    for ipa in ("ˈnaɡoja", "ljɔ̃", "ˈmʏnçn̩", "ʈʂɑŋ", "ˈdvɔr̝aːk", "ˈɡøːtə", "ˌeɪtiːənˈtiː", "ˈklɛə"):
        for sym in engines.cartesia(ipa).replace(">> <<", "|").strip("<>").split("|"):
            assert sym in engines._CARTESIA, (ipa, sym)
    with pytest.raises(ValueError, match="'ǃ'.*ǃxóõ"):
        engines.cartesia("ǃxóõ")
    table = {"Taa": Entry("Taa", "tah", ipa="ǃxóõ")}
    with pytest.raises(ValueError, match="^Taa: 'ǃ'"):                     # the script's word, then the symbol
        engines.to_engine("Taa is a language.", table, "cartesia")


def test_misaki_uses_its_own_symbols():
    assert engines.misaki("sɛrˈɡeɪ") == "sɛɹɡˈA"


def test_misaki_never_leaves_its_us_english_set():
    import pytest
    assert engines.misaki("ˈsɜːrʃə") == "sˈɜɹʃə" and engines.misaki("ˈwɔːtər") == "wˈɔtəɹ"   # no ɝ/ɚ in misaki
    assert engines.misaki("ˈmʏnçn̩") == "mˈʊnhən"              # ʏ isn't in Kokoro's vocab: it would be dropped
    assert engines.misaki("ljɔ̃") == "ljɔn"                    # a nasal vowel keeps its n
    assert engines.misaki("ˈɡəʊtə") == "ɡˈOtə"                 # British əʊ is the American O, not Q
    assert engines.misaki("ɕi tɕinpʰiŋ") == "ʃi ʧinpiŋ" and engines.misaki("bax") == "bɑk"
    for ipa in ("ˈnaɡoja", "ʈʂɑŋ", "ˈdvɔr̝aːk", "ˈɡøːtə", "t͡ʃaɪˈkɒf.ski", "ˈklɛə", "ʒɑ̃ lyk"):
        assert set(engines.misaki(ipa)) <= engines._MISAKI_US | {" "}, ipa
    with pytest.raises(ValueError, match="'ǃ'"):
        engines.misaki("ǃxóõ")


def test_piper_gets_only_what_espeak_en_us_writes():
    import pytest
    table = {"Bach": Entry("Bach", "bahk", ipa="bax"), "München": Entry("München", "x", ipa="ˈmʏnçn̩"),
             "Saoirse": Entry("Saoirse", "x", ipa="ˈsɜːrʃə"),
             "Kubernetes": Entry("Kubernetes", "x", ipa="ˌkuːbərˈnɛtiːz")}
    assert engines.to_engine("Bach, München.", table, "piper") == "[[ bɑːk ]], [[ mˈʊnhən ]]."   # no x, ʏ, ç
    assert engines.to_engine("Saoirse", table, "piper") == "[[ sˈɜːʃə ]]"          # espeak: nˈɜːs, no ɹ after ɜː
    assert engines.to_engine("Kubernetes", table, "piper") == "[[ kˌuːbɚnˈɛtiːz ]]"
    for ipa in ("ˈnaɡoja", "ʈʂɑŋ", "ˈɡøːtə", "ɕi tɕinpʰiŋ", "t͡ʃaɪˈkɒf.ski", "ljɔ̃"):
        assert set(engines._espeak(ipa)) <= engines._ESPEAK_US | {" "}, ipa
    with pytest.raises(ValueError, match="^Taa: 'ǃ'"):
        engines.to_engine("Taa", {"Taa": Entry("Taa", "tah", ipa="ǃxóõ")}, "piper")


def test_pls():
    x = engines.pls(TABLE)
    assert "<grapheme>Szegedy</grapheme>" in x and "<phoneme>ˈsɛɡɛdi</phoneme>" in x
    assert "<alias>rel-you</alias>" in x and 'alphabet="ipa"' in x


def test_pls_writes_a_phrase_as_one_lexeme_per_word():
    # Azure: "You can't directly set the pronunciation of a phrase", and no white space inside <phoneme>.
    table = {"Xi Jinping": Entry("Xi Jinping", "shee jin-ping", ipa="ɕi tɕinpʰiŋ"),
             "Xi": Entry("Xi", "she", ipa="ʃiː"),                             # its own entry wins over the phrase's
             "New York": Entry("New York", "noo york", ipa="nuːˈjɔːrk"),      # can't be split: the respelling
             "Sergey Ioffe": Entry("Sergey Ioffe", "Sair-gay Yoff-ee"),
             "Jean-Luc": Entry("Jean-Luc", "zhahn-look", ipa="ʒɑ̃ lyk")}
    x = engines.pls(table)
    assert "<grapheme>Xi Jinping</grapheme>" not in x
    assert "<grapheme>Jinping</grapheme>\n    <phoneme>tɕinpʰiŋ</phoneme>" in x
    assert x.count("<grapheme>Xi</grapheme>") == 1
    assert "<phoneme>ʃiː</phoneme>" in x and "<phoneme>ɕi</phoneme>" not in x
    assert "<grapheme>New York</grapheme>\n    <alias>noo york</alias>" in x
    assert "<grapheme>Sergey Ioffe</grapheme>\n    <alias>Sair-gay Yoff-ee</alias>" in x
    assert "<grapheme>Jean-Luc</grapheme>\n    <phoneme>ʒɑ̃.lyk</phoneme>" in x
    import re
    assert not any(" " in ph for ph in re.findall(r"<phoneme>(.*?)</phoneme>", x))


def test_scan_flags_names_acronyms_and_skips_the_lexicon():
    out = scan("Szegedy and Sergey Ioffe used ReLU in PyTorch with GPT-4o. The model is fast.", TABLE, rarity=False)
    words = {e["word"]: e["reasons"] for e in out["research"]}
    assert "Sergey" in words and "Ioffe" in words and "PyTorch" in words and "GPT-4o" in words
    assert "Szegedy" not in words and "ReLU" not in words and "The" not in words and "model" not in words
    assert {e["word"] for e in out["in_lexicon"]} == {"Szegedy", "ReLU"}


def test_scan_skips_inline_markup():
    out = scan("In {1959|nineteen fifty-nine} the {PDP-1|P D P one} ran.", {}, rarity=False)
    assert out["research"] == []


def test_scan_flags_a_rare_name_at_the_start_of_a_sentence():
    import pytest
    from voicecheck.scan import _tokenizer
    if _tokenizer() is None:
        pytest.skip("Whisper's tokenizer isn't available offline")
    words = {e["word"] for e in scan("Joaquin cut it. Batch norm came later. Ask Christian.", {})["research"]}
    assert "Joaquin" in words and "Batch" not in words
