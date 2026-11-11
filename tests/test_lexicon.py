import pytest

from voicecheck import config, lexicon
from voicecheck.lexicon import Entry


TABLE = {
    "Szegedy": Entry("Szegedy", "Seg-eh-dee", ipa="ˈsɛɡɛdi", source="https://en.wikipedia.org/wiki/Christian_Szegedy"),
    "ReLU": Entry("ReLU", "rel-you"),
    "AI": Entry("AI", "A I"),
    "tanh": Entry("tanh", "tanch"),
    "Sergey Ioffe": Entry("Sergey Ioffe", "Sair-gay Yoff-ee"),
}


def test_render_spoken_shown_and_ssml():
    r = lexicon.render("Szegedy's ReLU & AI use Tanh. Ask Sergey Ioffe.", TABLE)
    assert r.spoken == "Seg-eh-dee's rel-you & A I use tanch. Ask Sair-gay Yoff-ee."
    assert r.shown == "Szegedy's ReLU & AI use Tanh. Ask Sergey Ioffe."
    assert '<phoneme alphabet="ipa" ph="ˈsɛɡɛdiz">Szegedy&#x27;s</phoneme>' in r.ssml     # the 's is in the IPA
    assert r.terms[0].ipa == "ˈsɛɡɛdiz"
    assert '<sub alias="rel-you">ReLU</sub>' in r.ssml and "&amp;" in r.ssml
    assert [t.key for t in r.terms] == ["Szegedy", "ReLU", "AI", "tanh", "Sergey Ioffe"]


def test_capitalised_keys_are_case_sensitive():
    r = lexicon.render("Thai food said ai, not AI.", TABLE)
    assert r.spoken == "Thai food said ai, not A I."


def test_inline_markup_wins():
    r = lexicon.render("{ReLU|ray-loo} then ReLU", TABLE)
    assert r.spoken == "ray-loo then rel-you" and r.shown == "ReLU then ReLU"
    assert r.terms[0].key is None


def test_locate_finds_each_occurrence():
    r = lexicon.render("Szegedy met Szegedy", TABLE)
    words = ["seg", "eh", "dee", "met", "seg", "eh", "dee"]
    assert [lexicon.locate(words, t) for t in r.terms] == [(0, 3), (4, 7)]


def test_load_merges_settings_and_files(tmp_path):
    f = tmp_path / "lex.yaml"
    f.write_text("Szegedy:\n  say: Seg-eh-dee\n  ipa: ˈsɛɡɛdi\nReLU: ray-loo\n", encoding="utf-8")
    config.reset({"pronunciations": {"ReLU": "rel-you", "XOR": "ex-or"}})
    t = lexicon.load([f])
    assert t["ReLU"].say == "ray-loo" and t["XOR"].say == "ex-or" and t["Szegedy"].ipa == "ˈsɛɡɛdi"


def test_bad_entry_is_explained(tmp_path):
    f = tmp_path / "lex.yaml"
    f.write_text("Szegedy:\n  source: somewhere\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs `say`"):
        lexicon.load([f], base={})


# ---- bug hunt 2026-11-11

NGUYEN = {"Nguyen": Entry("Nguyen", "Win", ipa="wɪn")}


def test_locate_skips_an_ordinary_word_spelled_like_the_respelling():
    from voicecheck.text import words
    r = lexicon.render("The win was big, said Nguyen. Nguyen will win.", NGUYEN)
    spoken = words(r.spoken)                               # the win was big said win win will win
    assert [lexicon.locate(spoken, t) for t in r.terms] == [(5, 6), (6, 7)]         # was (1, 2), (5, 6)
    r = lexicon.render("Rel you later. ReLU is ReLU.", TABLE)
    assert [lexicon.locate(words(r.spoken), t) for t in r.terms] == [(3, 5), (6, 8)]


def test_terms_know_where_they_stand_in_the_shown_text():
    r = lexicon.render("The MAIN idea of AI. {ReLU|ray-loo} then Szegedy's ReLU", TABLE)
    assert [t.shown for t in r.terms] == ["AI", "ReLU", "Szegedy's", "ReLU"]
    assert all(r.shown[t.at:t.at + len(t.shown)] == t.shown for t in r.terms)
    assert r.terms[0].at == r.shown.index(" AI") + 1       # not the AI inside MAIN


def test_hyphenated_compounds_plurals_and_either_apostrophe():
    r = lexicon.render("a Szegedy-style net, pre-AI and AI-based", TABLE)
    assert r.spoken == "a Seg-eh-dee-style net, pre-A I and A I-based"
    assert [t.key for t in r.terms] == ["Szegedy", "AI", "AI"]
    table = {"GPU": Entry("GPU", "jee-pee-you", ipa="ˌdʒiːpiːˈjuː"), "AI": Entry("AI", "A I"),
             "O'Brien": Entry("O'Brien", "oh-BRY-un")}
    r = lexicon.render("Two GPUs, the GPU's fan, many AIs, MAIN, GPUS. O’Brien's", table)
    assert r.spoken == "Two jee-pee-yous, the jee-pee-you's fan, many A I's, MAIN, GPUS. oh-BRY-un's"
    assert r.shown == "Two GPUs, the GPU's fan, many AIs, MAIN, GPUS. O’Brien's"
    assert r.terms[0].shown == "GPUs" and r.terms[0].ipa == "ˌdʒiːpiːˈjuːz"          # the plural s is in the IPA


def test_yaml_values_that_are_not_text_are_refused(tmp_path):
    def load(text):
        f = tmp_path / "lex.yaml"
        f.write_text(text, encoding="utf-8")
        return lexicon.load([f], base={})
    for bad in ["ReLU:\n", "X: true\n", "A: [x]\n", "A: {say: yes}\n", "A: {ipa: 5}\n", "A: ''\n"]:
        with pytest.raises(ValueError, match="must be text"):                       # was the voice text "None", "True"
            load(bad)
    with pytest.raises(ValueError, match="in quotes"):
        load("NO: no way\n")                                                        # YAML reads the key NO as False
    with pytest.raises(ValueError, match="not valid YAML"):
        load("A: [unclosed\n")
    assert load("1984: nineteen eighty-four\n\"NO\": \"no\"\n")["1984"].say == "nineteen eighty-four"
    assert load("﻿Szegedy: Seg-eh-dee\n")["Szegedy"].say == "Seg-eh-dee"       # Notepad's byte-order mark
    assert lexicon.render("In 1984", lexicon.load([], base={1984: "nineteen eighty-four"})).spoken == \
        "In nineteen eighty-four"                                                   # was an AttributeError
    with pytest.raises(ValueError, match="expected a mapping"):
        lexicon.load([], base=["a", "b"])


def test_an_entry_with_only_ipa_is_respelled_for_a_plain_text_voice():
    e = lexicon._entry("Szegedy", {"ipa": "ˈsɛɡɛdi"})
    assert e.say == "Seg-eh-dee" and e.ipa == "ˈsɛɡɛdi"                             # was say="Szegedy"
    assert lexicon.render("Ask Szegedy.", {"Szegedy": e}).spoken == "Ask Seg-eh-dee."
    assert lexicon._entry("relu", {"ipa": "ˈrɛljuː"}).say == "rel-yoo"              # a lower-case key stays lower
    e = lexicon._entry("Nietzsche", {"ipa": "/ˈniːtʃə, -tʃi/"})                     # a source's two variants
    assert e.ipa == "ˈniːtʃə" and e.say == "Nee-chuh"


def test_possessive_after_other_sibilants_and_voiceless_sounds():
    assert lexicon.possessive("ˈlaɪbnɪts") == "ˈlaɪbnɪtsɪz"                         # was ...tsz
    assert lexicon.possessive("bɑːx") == "bɑːxs" and lexicon.possessive("ɪç") == "ɪçs"
    assert lexicon.possessive("wɪn") == "wɪnz" and lexicon.possessive("/kræft/") == "/kræfts/"
