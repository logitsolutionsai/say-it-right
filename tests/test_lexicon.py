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
