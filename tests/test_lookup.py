"""Wikitext parsing, offline (the live API calls are exercised by hand; see README)."""
import hashlib

from voicecheck.lookup import _NAME_FILE, _about, _fold, _ipac, commons_url, records, templates

SAOIRSE = ("'''Saoirse Una Ronan'''{{efn|({{IPAc-en|ˈ|s|əːr|ʃ|ə|_|ˈ|uː|n|ə|_|ˈ|r|oʊ|n|ə|n}} {{respell|SUR|shə|_|OO|"
           "nə|_|ROH|nən}}; {{IPA|ga|ˈsˠiːɾˠʃə|}})}} (born 12 April 1994) is an actress.")
LECUN = "'''Yann André LeCun''' ({{IPAc-en|l|ə|ˈ|k|ʌ|n}} {{respell|lə|KUN}}, {{IPA-fr|ləkœ̃|lang}}) is a scientist."
ZAREMBA = "'''Wojciech Zaremba''' ({{IPAc-pl|w|o|j|ć|e|ch|_|z|a|r|e|m|b|a}}) is a Polish computer scientist."
IOFFE = "'''Abram Ioffe''' ({{lang-rus|Абра́м Фёдорович Ио́ффе|p=ɐˈbram ˈfʲɵdərəvʲɪtɕ ɪˈofɛ}}) was a physicist."


def ipas(t):
    return [{k: v for k, v in x.items() if k != "raw"} for x in t["ipa"]]


def test_ipac_en_inside_a_footnote_with_aliases_resolved():
    t = templates(SAOIRSE)
    assert {"lang": "en", "ipa": "ˈsɜːrʃə ˈuːnə ˈroʊnən", "template": "IPAc-en"} in ipas(t)   # əːr is an alias
    assert {"lang": "ga", "ipa": "ˈsˠiːɾˠʃə", "template": "IPA"} in ipas(t)
    assert t["respell"] == ["SUR-shə OO-nə ROH-nən"]


def test_old_style_and_lang_templates():
    assert {"lang": "fr", "ipa": "ləkœ̃", "template": "IPA-fr"} in ipas(templates(LECUN))
    assert {"lang": "en", "ipa": "ləˈkʌn", "template": "IPAc-en"} in ipas(templates(LECUN))
    assert {"lang": "rus", "ipa": "ɐˈbram ˈfʲɵdərəvʲɪtɕ ɪˈofɛ", "template": "lang-rus"} in ipas(templates(IOFFE))


def test_ipac_in_other_languages_is_spelling_not_ipa():
    assert templates(ZAREMBA)["ipa"] == []


def test_ipac_labels_are_not_phonemes():
    assert _ipac(["uk", "ˈ", "s", "ɛ", "ɡ", "ə", "d", "i"]) == "ˈsɛɡədi"


def test_audio_files_and_direct_urls():
    t = templates("{{IPA|hu|ˈsɛɡɛd|lang|hu-Szeged.ogg}} {{Audio|En-us-LeCun.ogg|listen}} {{IPAc-en|w|ɪ|n|audio=x.oga}}")
    assert t["audio"] == ["hu-Szeged.ogg", "En-us-LeCun.ogg", "x.oga"]
    h = hashlib.md5("Hu-Krisztián.ogg".encode()).hexdigest()
    assert commons_url("File:Hu-Krisztián.ogg") == \
        f"https://upload.wikimedia.org/wikipedia/commons/{h[0]}/{h[:2]}/Hu-Kriszti%C3%A1n.ogg"
    assert commons_url("Hu-Krisztián.ogg").endswith("/0/02/Hu-Kriszti%C3%A1n.ogg")       # verified on Commons


def test_name_recording_filenames():
    assert _NAME_FILE.match("LL-Q150 (fra)-Arthur Crbz-André.wav")
    assert _NAME_FILE.match("Hu-Krisztián.ogg") and _NAME_FILE.match("En-us-Joaquin.oga")
    assert not _NAME_FILE.match("Speech by Joaquin Phoenix at the Oscars.ogg")
    assert records("En-us-Joaquin.oga", "Joaquin") and records("LL-Q9142 (gle)-Wikiroisin-saoirse.wav", "Saoirse")
    assert records("LL-Q150 (fra)-Fabricio Cardenas (Culex)-Emmanuel Macron.wav", "Macron")
    assert not records("En-us-anti-Christian.ogg", "Christian")
    assert not records("De-Christian Wolfgang Lindner.ogg", "Christian")


def test_search_hits_must_be_about_the_name():
    assert _about("Saoirse Ronan", "Saoirse Ronan")
    assert not _about("Wojciech Zaremba", "Christian Szegedy")
    assert not _about("Abram Ioffe", "Sergey Ioffe") and _about("Abram Ioffe", "Sergey Ioffe", person=False)
    assert _fold("Iоffe") == _fold("Ioffe")               # Wikidata's label uses a Cyrillic о
