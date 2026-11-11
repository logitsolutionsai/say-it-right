# What `vc.py proof` does

`check` trusts the script and judges the audio. `proof` trusts the audio and judges a transcript: one that Whisper,
a meeting notetaker, an auto-captioner or a typist wrote. It answers four questions about it, in this order.

## 1. Did the transcript write the names in the lexicon right?

The lexicon (`--lexicon`) is the list of names and terms that matter, with their spelling and, ideally, their IPA
(the same file `check` uses; research them the same way).

- **A name written right** is left alone, and when it has `ipa` the audio at that spot is held to it. If it
  doesn't sound like the name, that is a `name-mismatch` (the transcript wrote Siobhan, the speaker said Shannon).
- **A name said but written another way.** Two searches find those places. The transcript's own near-spellings
  (letters at least 65% like the name or its respelling: "Shivon", "Segedi") are checked by sound. And for a name
  of five sounds or more, its IPA is searched for in the phones of the whole recording (`ipa.spot`, the same
  distance `check` uses for names), which finds "Chevonne" and "she von" too. Where the audio matches the IPA and
  the transcript has other words, the issue is `name` and the fix is the lexicon's spelling. A near-spelling that
  sounds right is `high`; a sound match alone is `medium`, or `low` when the transcript's own words also fit the
  audio there (a homophone: "win" for Nguyen), because only the meaning can decide that.
- **A lexicon entry with no `ipa`** is still matched by spelling, and its respelling is aligned to the audio in
  place of what the transcript wrote; if it fits about as well, the issue is `name` at `medium`.

Short names (under five sounds) are only looked for where the spelling is close: /wɪn/ is in every "when" and
"window".

## 2. Does a second transcript disagree, and which one does the audio support?

By default Whisper (`whisper-small.en`) transcribes the audio again; `--second FILE` uses another tool's transcript
instead, and `--no-second` skips this step. The two texts are compared word by word, numbers and abbreviations in
whichever reading matches ("1959" and "nineteen fifty-nine" are the same words).

Where they differ, neither is believed. Both wordings are force-aligned to the same stretch of audio, between the
words either side that the two agree on, by the letter model, which has no language model and so no opinion about
what would make a good sentence. The difference in log-likelihood (`letters_gain`) is the evidence:

- the second transcript's wording fits better by `proof_gain` (4) or more: `misheard` (a word swapped), `missing`
  (the transcript skipped words), or `unheard` (the transcript has words the other doesn't, and the audio fits
  better without them);
- the transcript's own wording fits better: nothing is reported. A second opinion that is wrong costs nothing;
- neither clearly: `unsure`, only if the transcript's word also scores low.

A `misheard` is `high` when the transcript's word scores under 0.65 against the audio and the two wordings are
clearly different words (or numbers); when they are spelled alike (affect/effect, there/their) it is `medium` and
says so, because sound can't separate homophones. A swap between two words of three letters or fewer (a/the,
in/of) is `high` only on a wide margin (five times `proof_gain`): in fluent speech those blur, and on LibriSpeech
they were where the wrong `high` fixes came from.

**Names are the exception.** The letter model reads spelling, and a name's spelling doesn't follow its sound:
"Shivon" fits the audio of *Siobhan* better than "Siobhan" does. So when either wording of a disagreement is a
name (capitalised mid-sentence, an acronym, or a capitalised word the tokenizer finds rare) the issue is
`spelling`, with no fix offered: find out the right spelling, put it in the lexicon, and step 1 handles it.

## 3. Is there text that isn't in the audio?

The whole transcript is aligned to the audio by the letter model. A run of words it can't find (letter score under
0.25) is `unheard`. If nothing at all is said in their place, the fix is to delete them: at `high` for three words
or more (this is what a hallucinated closing line looks like), at `medium` for one or two short words. If
something else is said there, the issue says what the letter model hears instead; for a single name or rare word
it is `spelling`, because a name's letters often aren't found even when it is said.

Text that isn't in the audio has to be put somewhere, and an alignment pays for it with the words next to it
(after "...until we are gone. Please subscribe to the channel", the words "we are" went missing too). So the
words that can't be found are taken out and the rest is aligned a second time: they stay unheard, their
neighbours come back. And before a run is reported, any of its words that the letter model does hear in that
stretch, in order, are taken back out of it.

## 4. Is there speech that isn't in the text, or a word that sounds like another?

With no script at all, the letter model writes what it hears (rough spelling, nothing "corrected").

- Words it hears between two aligned transcript words, or before the first and after the last, are `missing`.
  Fillers (uh, um) are ignored; a two- or three-letter word counts only when every letter is clear. Audio before
  and after the transcript is not smeared over its first and last words: a wildcard at each end absorbs it.
- Where it hears a different word from the transcript's, both are aligned to the same stretch. If what it heard
  fits better by `proof_heard_gain` (10) or more, the word is reported: `misheard` at `medium`, with what was heard
  as the suggestion, when the gain is 25 or more; `unsure` below that; `spelling` when the word is a name or a rare
  word. A suggestion from this step is spelled by ear and is never `high`.

This step is the one that still works when both transcripts came from models with language models and both
"corrected" the same slip.

## Confidence, and what `--write` applies

| Confidence | Meaning | `--write` |
|---|---|---|
| `high` | two kinds of evidence agree (a second transcript and the letters; a near-spelling and the phones; no speech at all under the words) | applied by default |
| `medium` | one kind of clear evidence; the `why` says what to check | with `--apply medium` |
| `low` | worth a listen; often no fix is offered | with `--apply all`, where there is one |

A fix keeps the transcript's format: SRT and WebVTT keep their numbering, timing lines and tags (an edit that
crosses two captions puts its text in the first); Whisper JSON gets each segment's `text` rewritten (and its stale
`words` list removed); plain text and Markdown keep their punctuation and line breaks. Speaker labels at the start
of lines ("Maria:", "SPEAKER_01:"), `[Music]`, `(laughs)`, `<i>` tags and time stamps are not treated as speech.

When the letter model hears fewer than 30% of the transcript's words anywhere in the audio (`agreement`), the
report says the transcript looks like another recording's, and every issue drops to `low`.

## Report fields

`clean`, `issues` (each: `kind`, `at` [seconds], `line`, `cue`, `transcript`, `suggest`, `confidence`, `why`,
`context`, `evidence`, `span` and `edit` [start, end, text] in characters of the transcript), `counts` by kind,
`words`, `lexicon` (per entry: `written`, `confirmed` by sound, `doubted`, `restored`), `agreement`, `aligned`,
`weak_words` (found, but scoring under 0.65), `second_opinion`, `note`, `timing`, and `written` after `--write`.

## Measured (2026-11-11)

`benchmarks/proof_librispeech.py`: 150 utterances of LibriSpeech test-clean (real people reading; 33.6 minutes,
5,513 words), each with one injected error (25 each of: a word swapped for an unrelated one, for a similar one,
one or two words dropped, one or two added, a sentence appended) or none (25).

| | With Whisper as second opinion | Letter ear alone |
|---|---|---|
| Injected errors found | 125 / 125 | 123 / 125 |
| Word error rate before | 4.39% | 4.39% |
| after the `high` fixes | 0.65% | 2.54% |
| after `high` + `medium` | 0.83% | 1.41% |
| Wrong `high` fixes | 1 in 5,513 words | 0 |
| Issues not at the injected error, per 1,000 words (`high` / `medium` and up / all) | 0.18 / 4.35 / 6.17 | 0.00 / 1.09 / 3.81 |
| Untouched transcripts with nothing at `medium` or `high` | 21 / 25 | 25 / 25 |

Most `medium` issues away from the injected errors are Whisper's modern spelling against the book's ("removed"
for "remov'd", "taboo" for "tabu") and homophones ("red" for "read"), which is what `medium` is for.

Calibration of `proof_heard_gain` (60 other utterances of the same set, one substituted word in each, 2026-11-11):
every one of the 60 substitutions was heard differently by the letter model, with a gain of 11.6 or more (most
over 25). Of the correctly transcribed words it heard differently (14 in about 1,100), all but one gained 4.1 or
less; the exception was a rare name ("Whittaws", 12.6), which the name rule now turns into `spelling`. The word's
own score separates the two far worse: similar-word substitutions scored 0.5 to 1.0.

These are injected errors on read speech, and the letter model was trained on LibriSpeech's training sets: treat
the numbers as an upper bound for other audio. The name steps are covered by planted runs with synthetic voices
(`tests/test_models.py`), not by this benchmark.

## Limits

- English, like the rest of the checker.
- Homophones can't be told apart by sound; `medium` fixes between words spelled alike need a reader.
- The letter model was trained on read audiobooks. On clean read or presented speech it is near its best; on
  noisy, overlapping, heavily accented or very fast speech it mishears more, and the report will carry more
  `unsure` and `low` issues. `high` needs two kinds of evidence, so it degrades last.
- Overlapping speakers are one stream to it: it can't say who said what, and it has no speaker labels of its own.
- It does not re-time captions. `at` is where the audio is; cue times are left as they were.
- Speed on a laptop CPU: about 0.4x the audio's length for the letter model, the same again for Whisper, and
  about 0.3x for the phone model when the lexicon has IPA. An hour of audio is roughly an hour of work, in memory
  that doesn't grow with the length.
