"""Benchmark the name check's core (spotting a mispronounced word) against human experts: speechocean762.

speechocean762 (Zhang et al., Interspeech 2021; Apache-2.0 on Hugging Face) has 5,000 read English utterances by 250
non-native speakers, half of them children; five experts scored every word (accuracy 0-10) and every phone (0
incorrect or missed, 1 heavy accent, 2 correct). We run the checker's ears on the 2,500-utterance test split, word by
word, exactly as `vc.py check` does for a lexicon word: the letter model aligns the text to find each word's window,
the phone model transcribes the window, and the transcription is compared with the word's canonical phones.

Methods, all on the same recognisers and windows, so only the comparison differs:
    ours      voicecheck.ipa.compare: feature-weighted, accent-tolerant (the shipped default)
    per       plain phone error rate (every difference costs 1), with the same free word edges
    letters   the letter model's word score (the "slips" ear)
    both      the mean of ours and letters, equal weights (needs a word whose spelling the letter model can
              read, so the shipped checker uses it for no word: names go by phones alone, other words by
              letters alone; it shows how much the two ears add to each other)
Metrics: Pearson correlation with the experts' word accuracy; ROC-AUC for words the experts marked wrong (a phone
averaged under 1.0, i.e. most experts said incorrect or missed); precision / recall / F1 of the shipped pass/fail rule.

    python benchmarks/speechocean762.py [--limit N]        # resumable; results in benchmarks/results/
Needs `pip install pyarrow` (on Windows with Smart App Control, pyarrow==18.1.0 loads where 22+ is blocked).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voicecheck import config, ipa, verify  # noqa: E402
from voicecheck.phonemes import PhonemeEar, hard_errors, word_window  # noqa: E402

OUT = ROOT / "benchmarks" / "results"
ARPA = {"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
        "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i", "JH": "dʒ", "K": "k",
        "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
        "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"}


def to_ipa(arpa: list[str]) -> str:
    """ARPAbet -> IPA. Words of 2+ syllables keep their stress (ˈ before a stressed vowel, . before the others), as a
    lexicon entry would; one-syllable words get none (the dataset marks even "MARK" as AA0)."""
    vowels = [p for p in arpa if p[-1] in "012"]
    marked = len(vowels) > 1
    out = []
    for p in arpa:
        base = p.rstrip("012")
        sym = "ə" if (base == "AH" and p.endswith("0")) else "ɚ" if (base == "ER" and p.endswith("0")) else ARPA[base]
        if marked and p[-1] in "012":
            out.append("ˈ" if p.endswith("1") else ".")
        out.append(sym)
    return "".join(out)


FUNCTION = set("""a an the and or but if of to in on at by for from with as is are was were be been am do does did
have has had i you he she it we they me him her us them my your his its our their this that these those there here
not no so than then too very can will would shall should could may might must what who which when where how why""".split())


def per_score(target: str, heard: list[str]) -> float:
    """1 - phone error rate, with the same free edges as compare() (the window holds bits of the neighbours)."""
    t = [u[0] for u in ipa.units(target)]
    h = [u[0] for u in ipa.units(heard)]
    if not t:
        return 1.0
    prev = [0] * (len(h) + 1)
    for i in range(1, len(t) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j - 1] + (t[i - 1] != h[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return max(0.0, 1 - min(prev) / len(t))


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def auc(score_bad, labels) -> float:
    """ROC-AUC that a mispronounced word gets a lower score than a good one (Mann-Whitney U, ties count half)."""
    s, lab = np.asarray(score_bad, float), np.asarray(labels, bool)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(s):                                       # average ranks over ties
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos, neg = lab.sum(), (~lab).sum()
    # low score = mispronounced: AUC of -score for the positive (bad) class
    r_bad = ranks[lab].sum()
    u = r_bad - pos * (pos + 1) / 2
    return float(1 - u / (pos * neg))


def run(limit: int | None) -> Path:
    lis, ear = verify.Listener(), PhonemeEar()         # torch before pyarrow: on Windows, pyarrow's C++ runtime
    import pyarrow.parquet as pq                        # loaded first makes torch's c10.dll fail (WinError 1114)
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("mispeech/speechocean762", "data/test-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(path)
    n = table.num_rows if limit is None else min(limit, table.num_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    rows_path = OUT / "speechocean762_words.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["utt"])
    v = config.voice()
    t_start, processed = time.time(), 0
    with open(rows_path, "a", encoding="utf-8") as f:
        for k in range(n):
            row = table.slice(k, 1).to_pylist()[0]
            utt = row["audio"]["path"]
            if utt in done:
                continue
            y, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            scores = lis.word_scores(y, sr, row["text"])
            words = row["words"]
            if not lis.measured or len(lis.words) != len(words):
                f.write(json.dumps({"utt": utt, "skipped": "alignment" if not lis.measured else "word count"}) + "\n")
                continue
            for i, w in enumerate(words):
                target = to_ipa(w["phones"])
                win = word_window(lis.times, i, i + 1, len(y) / sr)
                heard = ear.hear(y, sr, max(0.0, win[0] - 0.03), win[1] + 0.03) if win else []
                c = ipa.compare(target, heard)
                hard = hard_errors(c.diff)
                f.write(json.dumps({
                    "utt": utt, "speaker": row["speaker"], "age": row["age"], "word": w["text"],
                    "human_accuracy": w["accuracy"], "phone_accuracy": w["phones-accuracy"],
                    "bad": bool(min(w["phones-accuracy"]) < 1.0), "target": c.target, "heard": " ".join(heard),
                    "ours": c.score, "hard": hard, "ours_fail": not (c.score >= v["phone_min"] and
                                                                     hard <= v["phone_hard_max"]),
                    "per": round(per_score(target, heard), 3), "letters": scores[i][1]}, ensure_ascii=False) + "\n")
            f.flush()
            processed += 1
            if processed % 25 == 0:
                rate = (time.time() - t_start) / processed
                print(f"{k + 1}/{n} utterances, {rate:.2f} s each, ~{rate * (n - k - 1) / 60:.0f} min left",
                      file=sys.stderr, flush=True)
    return rows_path


def summarise(rows_path: Path) -> dict:
    rows = [json.loads(x) for x in rows_path.read_text(encoding="utf-8").splitlines()]
    words = [r for r in rows if "word" in r]
    skipped = [r for r in rows if "skipped" in r]
    res = {"utterances": len({r["utt"] for r in rows}), "skipped_utterances": len(skipped), "words": len(words),
           "labels": {"incorrect_phone": "a phone the experts scored under 1.0 on average (most said incorrect or "
                                         "missed)", "low_word_accuracy": "experts' word accuracy 7 or less (of 10)"},
           "subsets": {}}
    for name, keep in (("all words", lambda r: True),
                       ("content words", lambda r: r["word"].lower() not in FUNCTION),
                       ("2+ syllable words", lambda r: r["target"].count(" ") >= 3 and
                        sum(c in "ɑæʌɔaɛɝɚeɪioʊuə" for c in r["target"].replace(" ", "")) >= 2)):
        g = [r for r in words if keep(r)]
        acc = [r["human_accuracy"] for r in g]
        sub = {"words": len(g), "incorrect_phone": int(sum(r["bad"] for r in g)),
               "low_word_accuracy": int(sum(r["human_accuracy"] <= 7 for r in g)), "methods": {}}
        for m in ("ours", "per", "letters", "both"):
            sc = [(r["ours"] + r["letters"]) / 2 if m == "both" else r[m] for r in g]
            sub["methods"][m] = {"pcc_word_accuracy": round(pearson(sc, acc), 3),
                                 "auc_incorrect_phone": round(auc(sc, [r["bad"] for r in g]), 3),
                                 "auc_low_word_accuracy": round(auc(sc, [r["human_accuracy"] <= 7 for r in g]), 3)}
        f_ = [r["ours_fail"] for r in g]
        b_ = [r["human_accuracy"] <= 7 for r in g]
        tp, fp = sum(a and b for a, b in zip(f_, b_)), sum(a and not b for a, b in zip(f_, b_))
        fn = sum(b and not a for a, b in zip(f_, b_))
        pr = tp / (tp + fp) if tp + fp else math.nan
        rc = tp / (tp + fn) if tp + fn else math.nan
        sub["shipped_rule_vs_low_word_accuracy"] = {"flagged": tp + fp, "precision": round(pr, 3),
                                                     "recall": round(rc, 3)}
        res["subsets"][name] = sub
    bad = [r["bad"] for r in words]
    fail = [r["ours_fail"] for r in words]
    tp = sum(f and b for f, b in zip(fail, bad))
    fp = sum(f and not b for f, b in zip(fail, bad))
    fn = sum(b and not f for f, b in zip(fail, bad))
    p = tp / (tp + fp) if tp + fp else math.nan
    r = tp / (tp + fn) if tp + fn else math.nan
    res["shipped_rule_vs_incorrect_phone"] = {"flagged": tp + fp, "precision": round(p, 3), "recall": round(r, 3),
                           "f1": round(2 * p * r / (p + r), 3) if p + r else math.nan}
    for group, keep in (("adults", lambda x: x["age"] >= 16), ("children", lambda x: x["age"] < 16)):
        g = [x for x in words if keep(x)]
        if g and any(x["bad"] for x in g) and not all(x["bad"] for x in g):
            res.setdefault("by_age", {})[group] = {
                "words": len(g), "auc_ours": round(auc([x["ours"] for x in g], [x["bad"] for x in g]), 3),
                "auc_per": round(auc([x["per"] for x in g], [x["bad"] for x in g]), 3)}
    (OUT / "speechocean762_summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N test utterances")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    path = OUT / "speechocean762_words.jsonl" if a.summary_only else run(a.limit)
    print(json.dumps(summarise(path), indent=2))
