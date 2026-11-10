from voicecheck import engines
from voicecheck.lexicon import Entry
from voicecheck.scan import scan

TABLE = {"Szegedy": Entry("Szegedy", "Seg-eh-dee", ipa="ˈsɛɡɛdi"), "ReLU": Entry("ReLU", "rel-you")}
LINE = "Szegedy used ReLU."


def test_text_and_ssml():
    assert engines.to_engine(LINE, TABLE, "text") == "Seg-eh-dee used rel-you."
    assert engines.to_engine(LINE, TABLE, "ssml") == (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        '<phoneme alphabet="ipa" ph="ˈsɛgɛdi">Szegedy</phoneme> used <sub alias="rel-you">ReLU</sub>.</speak>')
    azure = engines.ssml(LINE, TABLE, voice="en-US-AvaNeural")
    assert '<voice name="en-US-AvaNeural">' in azure and azure.endswith("</voice></speak>")
    assert engines.ssml(LINE, TABLE, bare=True).startswith("<speak><phoneme")


def test_possessives_carry_their_s_into_the_ipa():
    table = {"Nguyen": Entry("Nguyen", "Win", ipa="wɪn"), "Ruiz": Entry("Ruiz", "Roo-eess", ipa="ruˈiːs"),
             "Kraft": Entry("Kraft", "Kraft", ipa="kræft")}
    out = engines.ssml("Nguyen's, Ruiz's and Kraft's teams", table, bare=True)
    assert 'ph="wɪnz">Nguyen&#x27;s<' in out and 'ph="ɹuˈisɪz"' in out and 'ph="kɹæfts"' in out


def test_us_english_phones_for_azure_and_cartesia():
    assert engines.us_english("ˈsɜːrʃə") == "ˈsɝʃə"            # r-coloured, no length mark
    assert engines.us_english("ˈjofːɛ") == "ˈjoʊfɛ"            # a lone o is the English oʊ
    assert engines.us_english("ˈɑːndrɛj") == "ˈɑndɹeɪ"          # ɛj is the English eɪ, r is ɹ
    assert engines.us_english("ˈbaχ") == "ˈbak"


def test_inline_ipa_engines():
    assert engines.to_engine(LINE, TABLE, "elevenlabs") == "/ˈsɛɡɛdi/ used rel-you."
    assert engines.to_engine(LINE, TABLE, "cartesia") == "<<ˈ|s|ɛ|ɡ|ɛ|d|i>> used rel-you."
    assert engines.to_engine(LINE, TABLE, "kokoro") == "[Szegedy](/sˈɛɡɛdi/) used rel-you."
    assert engines.to_engine(LINE, TABLE, "piper") == "[[ sˈɛɡɛdi ]] used rel-you."


def test_cartesia_adds_the_required_stress():
    assert engines.cartesia("wɪn") == "<<ˈ|w|ɪ|n>>"


def test_misaki_uses_its_own_symbols():
    assert engines.misaki("sɛrˈɡeɪ") == "sɛɹɡˈA"


def test_pls():
    x = engines.pls(TABLE)
    assert "<grapheme>Szegedy</grapheme>" in x and "<phoneme>ˈsɛɡɛdi</phoneme>" in x
    assert "<alias>rel-you</alias>" in x and 'alphabet="ipa"' in x


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
