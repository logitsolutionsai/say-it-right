# What `vc.py check` measures

## The three ears

| Ear | Model | Why |
|---|---|---|
| Transcript | Whisper (`openai/whisper-small.en`) | Writes what was said, but its language model writes what it *expects*: a slip can come back "corrected". |
| Letters | wav2vec2 CTC (`facebook/wav2vec2-large-960h-lv60-self`), no language model | The script is force-aligned to the letters it hears; each word scores 0..1. A word that isn't in the audio can't hide. |
| Phones | wav2vec2 phoneme CTC (`facebook/wav2vec2-lv-60-espeak-cv-ft`, Apache-2.0) | For lexicon words with `ipa`: what sounds were actually made, against the target IPA. Spelling-free, so respellings and foreign names are judged fairly. |

## Report fields

`passed`, `reasons` (one line per failed check), `heard` (Whisper), `spoken` (what the voice was given), `shown`
(the on-screen wording), `match` + `diff`, `slips` (word and its letter score), `names` (per researched word:
`target`, `heard` (the part that lines up with the target), `around` (every phone in the window between the
neighbouring words), `score`, `hard`, `diff`, `t0`/`t1` in seconds), `listener_measured` (false when the audio is too
short to hold the script: every word then scores 0 and the take fails), `word_scores`, `syllables_per_s`,
`speech_seconds` (first word to last, so lead-in silence doesn't count), `tone`, `tone_flags`, `beams` (the
Whisper beam width that produced `heard`), `timing` (seconds spent loading models and on each ear).

## Rules and thresholds (all in `voicecheck/config.py`, overridable with `VOICECHECK_CONFIG`)

- `match_min` 0.95: letter similarity of transcript and script. Numbers and common abbreviations on either side
  are compared in the reading that fits the other ("1959" vs "nineteen fifty-nine", "100" vs "a hundred", "Dr." vs
  "doctor"), homophones count as equal (weight/wait), number-word homophones don't (to/two would hide "zero [to]
  two hundred").
- `beams` 1 / `retry_beams` 5: Whisper decodes greedily; only a transcript that misses `match_min` is decoded again
  with 5 beams (OpenAI's default), and that one decides. A passing take pays for one greedy pass (about half the
  time of beam search); a failing one pays for both. Measured on the 38 test takes: the same verdicts as always
  using 5 beams.
- `slip_score` 0.65 / `gone_score` 0.25: a word is a slip when Whisper's transcript differs there and its letter
  score is under 0.65, or when its score is under 0.25 whatever Whisper wrote. The letter model reads each number
  the way that best fits the audio (a year, a count, with or without "and", digit by digit).
- `phone_min` 0.6 and `phone_hard_max` 0: a name fails on any clear substitution (`!`) or missing sound, or a score
  under 0.6. Weak differences pass: unstressed vowel colour, an r after a vowel, cot/caught, a flapped d or t, a
  dropped schwa or glide. A wrong stressed vowel counts as clear (kar-PAH-thee said KAR-puh-thee fails).
- `pace_syl_s` (off by default) and `flat` limits `range_st` / `peak_st` / `tail_st` (off by default; the
  OpenMontage example sets 19 / 7.0 / 4.5 semitones). Pitch range auto-adapts to the voice (`pitch_hz: auto`).

## Calibration of the name check (2026-11-10)

32 takes: 8 names (Szegedy, Siobhan, Saoirse, Joaquin, Nguyen, Ioffe, Karpathy, Sergey) × 2 Windows voices
(male, female) × {read from the researched respelling, read from the raw spelling}. Several raw-spelling takes were
in fact right (the voices already knew Joaquin, and one said Sergey and Ioffe well); judged from the heard phones,
9 takes were wrong. With the defaults the check agreed on 32/32: every wrong take had at least one clear error
(soft-g "Zuh-jee-dee", "Yah-bun" for Siobhan, "Ser-jee", "Kar-PAY-thee", "KAR-puh-thee"), and no right take did.
This is a small, synthetic set, and the thresholds were chosen on it: expect some false alarms on unusual voices,
and read the `diff` before acting. For a held-out test of the same comparison on human speech, see the
speechocean762 benchmark in the README (`benchmarks/speechocean762.py`).

## Why these models and this distance

- **Phone model.** wav2vec2-lv-60-espeak-cv-ft (Apache-2.0, 1.26 GB) decodes with its `vocab.json` alone, so neither
  phonemizer nor espeak-ng is needed. Reported phone feature error rates (ZIPA paper, ACL 2025): 14.4 on seen
  languages, 3.5 on unseen/variation sets. **ZIPA-CR** (anyspeech/zipa-small-crctc-500k, CC BY 4.0, 71-260 MB ONNX)
  reports 5.6 / 3.7 and runs about 20x faster on a CPU; it is the natural upgrade, but needs onnxruntime and a
  Kaldi-style filterbank front end, and the thresholds here would need re-calibrating. Avoid the ZIPA "ns" and
  PhoneticXEUS checkpoints in commercial work (non-commercial licences); Allosaurus is GPL-3.0 and the least
  accurate. Any wav2vec2 CTC phone model with the same vocabulary works via the `phonemes` setting (e.g.
  facebook/wav2vec2-xlsr-53-espeak-cv-ft).
- **Distance.** `voicecheck/ipa.py` has its own feature-weighted edit distance rather than panphon's: panphon
  silently drops symbols it doesn't know (ASCII g), its `editdistance` dependency had no wheels for Python 3.13,
  and it pulls in pandas (whose DLL Windows Application Control blocked on the test machine). The local one is
  also accent-aware and stress-aware, which panphon's isn't.

## Long takes

A take of up to 40 seconds goes through the two wav2vec2 models whole. A longer one is heard in windows of 30
seconds, each with 5 seconds of audio either side whose frames are thrown away, and the windows are joined: the
models' memory grows with the square of the length, and a five-minute take heard whole needs more than 10 GB. The
script is then aligned in sections, cut where the script and the letter model's own hearing agree word for word
(at the word boundary the model marked), so the alignment table stays small. On the same posteriors, section-wise
and whole alignment give the same word scores and times (`tests/test_proof.py`, and a 130-second take: 369 of 369
words identical). Windowed hearing itself moves a few word scores slightly (one word in 369 by more than 0.2 on
that take); the verdict was the same. A stretch where the script and the audio share too little to cut (a script
for a different recording) is left unmeasured and the take fails with a reason that says where.

## Known limits

- **Stress is only heard through vowels.** Neither phone model outputs stress marks. A stress shift that changes a
  vowel is caught (kar-PAH-thee said KAR-puh-thee reduces the stressed ɑ to ə); one that keeps every vowel
  (LE-cun for lə-KUN, if the voice keeps the schwa) is not. Listen to names where stress is the whole point.

- **Short names are judged loosely.** A name of three sounds (/wɪn/ for Nguyen) has little to hold the audio to: another short word with close sounds can pass. Names of five sounds or more are where the check is strong.
- English scripts only (Whisper small.en and the letter model are English).
- Words with no `ipa` get only the transcript and letter checks; respelled pieces after the first aren't scored by
  letters (the "you" of rel-you is spoken "yu").
- The phone model hears a word inside a window found by the letter model; if the letter alignment is far off (a
  badly garbled take) the window is too, and the take usually fails on slips anyway.
