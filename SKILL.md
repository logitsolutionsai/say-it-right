---
name: say-it-right
license: MIT
description: Make AI voice-overs say every word right, especially people's names, and prove it before the audio ships; and the reverse: proofread a transcript against its audio. Researches how names and rare terms are pronounced (Wikipedia/Wiktionary/Wikidata IPA, recordings, web search), keeps them in a sourced lexicon, renders the script for the TTS engine (respellings, SSML phonemes, ElevenLabs/Cartesia/Kokoro/Piper IPA, PLS dictionaries), then checks each take with three ears (Whisper, a letter model with no language model, an IPA phone model) for slipped or missing words, mispronounced names, pace and tone. The same ears check any tool's transcript or captions against the recording: hallucinated text, skipped speech, misheard words, misspelled names. Use for narration, voice-over, podcast, audiobook or any TTS work, pronouncing a name, pronunciation dictionaries, verifying a take against its script, or checking or fixing a transcript or subtitles against the audio, even if nobody says "pronunciation" or "check".
---

# say-it-right

A TTS voice reads a script well until it meets a name ("Szegedy", "Siobhan", "Nguyen"), a number it must guess
how to say ("1959 people"), or a word whose spelling misleads it. It doesn't warn you: Whisper's transcript of the
take will even "correct" the mistake back to the spelling. This skill closes that gap in four moves:

1. **Find** the risky words in the script.
2. **Research** how each one is said, and record it with a source in a lexicon.
3. **Render** the script in the syntax the user's TTS engine understands.
4. **Check** every take, and re-record or re-spell until it passes.

The same ears also work in reverse (section 5): given a recording and a transcript some tool wrote of it, they find
the text that isn't in the audio, the speech the transcript skips, the misheard words and the misspelled names.

Everything runs from `vc.py` in this skill's folder: `${CLAUDE_SKILL_DIR}` (in agents that don't substitute it, the
folder holding this file). Below, `vc` stands for `python "${CLAUDE_SKILL_DIR}/vc.py"` (`python3` on macOS and
Linux); any Python works, vc.py runs itself inside the skill's own environment. Results are JSON on stdout,
progress on stderr; exit code 0 = passed, 1 = a take failed a check (or `proof` found issues), 2 = the command
failed.

## Setup (once)

```bash
python "${CLAUDE_SKILL_DIR}/install.py"          # creates the skill's environment; ~1 minute with uv, a few without
python "${CLAUDE_SKILL_DIR}/install.py" --check  # later: is it still installed, and where?
```

If a command says the skill isn't installed, or that it was updated, run the first line. The first `check`
downloads Whisper small.en and wav2vec2 (about 2.2 GB) and the first check of a researched name adds the phoneme
model (1.3 GB); add `--models` to install.py to download them up front. All run on the CPU.

## 1. Find the words that need research

```bash
vc scan script.txt --lexicon lexicon.yaml
```

`research` lists names (capitalised mid-sentence), acronyms, mixed-case terms (PyTorch), letters outside English
(Søren), letters glued to digits (GPT-4o), and words rare enough that the TTS will guess (several tokenizer pieces;
this catches a name at the start of a sentence). `numbers` shows how each number could be read. Also read the
script yourself: you will know names the heuristics miss (a common-looking word that is a surname, a brand).

For an acronym, decide whether it is said as letters (A I, G P T) or as a word (NASA, ReLU), and for an ambiguous
number either spell it out in the script ("nineteen fifty-nine") or accept that the checker tries every reading.

## 2. Research each name

Work from the most authoritative source down, and stop when a source is clearly authoritative:

1. **The person or owner themselves**: a video or podcast where they say their own name, a "how to pronounce my
   name" note, a name-pronunciation field on a profile, a company's own ad saying its brand. Use web search for
   `"<name>" pronounce`, `"<name>" interview`, `how do you say <name>`.
2. **Recordings by native speakers**: Wikidata/Lingua Libre/Commons audio (`lookup` finds them). Turn one into IPA
   with `vc hear <url-or-file>` and compare it with what the text sources say.
3. **Reference IPA**: Wikipedia's lead-section IPA and respelling, Voice of America's pronunciation guide,
   Wiktionary. `vc lookup "Full Name" --context "what they are known for"` collects these, plus the
   recordings above, from keyless public APIs in one go. Pages marked `match: "surname only"` are about someone
   else with the same surname: useful for how the surname is said, not proof of how this person says it. Sources
   it couldn't reach are listed in `errors`; an empty result is common (most people have no recorded
   pronunciation anywhere), which is exactly when the web search in step 1 matters.
4. **The language's spelling rules** (Hungarian sz = s, Irish bh = v, Polish ł = w), stated as such.

Decide the *target* before writing anything down: the person's own pronunciation wins; otherwise use the
established English form if one exists (BBC/broadcast practice: anglicise only as far as is customary), otherwise
the native form approximated with English sounds. When the sources disagree and none is the person's own, say so
to the user and pick the best-supported one, or ask if it matters (a guest's name on their own show does).
`references/name-research.md` has the source details, the API calls and worked examples.

Record each decision in a lexicon file, with where it came from:

```yaml
# lexicon.yaml
Szegedy:
  say: Seg-eh-dee          # what a plain-text TTS reads
  ipa: ˈsɛɡɛdi             # the target sound; the phoneme check holds every take to it
  source: Hungarian spelling (sz=s, final y=i); English-language talks say SEG-eh-dee
Siobhan: {say: Shiv-awn, ipa: ʃɪˈvɔːn, source: "https://en.wikipedia.org/wiki/Siobhan"}
ReLU: rel-you              # a bare respelling: no IPA, so only the transcript and letter checks apply
AI: A I                    # an acronym said as letters
```

Always give `ipa` for a person's name: it is what lets the checker hear whether the name was said right. Get the
respelling from the IPA with `vc respell "ʃɪˈvɔːn" --name`, which follows Wikipedia's respelling key and
returns a few candidates (plain `Shiv-awn`, stress-marked `Shiv-AWN`, spaced); `references/respelling.md` has the key.

## 3. Render the script for the engine

```bash
vc render script.txt --lexicon lexicon.yaml -f <format>
```

| Engine | Format | What it gets |
|---|---|---|
| Qwen3-TTS, Chatterbox, Kyutai Pocket, Gradium Phonon, Coqui XTTS, Sesame CSM, Dia, OpenAI TTS, ElevenLabs Multilingual v2 / Flash v2.5 | `text` | respellings in plain text |
| Azure (add `--voice <name>`); Google Cloud incl. Chirp 3 HD sync and Amazon Polly (add `--bare`, no `--voice`) | `ssml` | `<phoneme alphabet="ipa">` in US-English phones, every syllable marked, one per word; `<sub alias>` |
| ElevenLabs v4, Inworld TTS-2 | `elevenlabs` | inline `/IPA/` for researched words |
| ElevenLabs v3 / v4 / v4 Turbo / Flash v2, Azure `<lexicon>`, Polly lexicons | `export-pls` | a PLS dictionary file to upload |
| Cartesia Sonic 3.x | `cartesia` | `<<s\|ˈ\|ɛ\|...>>` inline IPA |
| Kokoro | `kokoro` | `[word](/misaki phones/)` |
| Piper | `piper` | `[[ espeak phonemes ]]` |

Engine details and caveats (which ElevenLabs models ignore phonemes, Chirp's sync-only SSML, Kokoro silently
dropping unknown symbols), and a dozen more engines with no format of their own yet (Deepgram, MiniMax, Fish Audio,
Rime, Murf...), are in `references/tts-engines.md`. If unsure of the engine, ask; `text` works everywhere. A sound
the engine has no symbol for stops `render` with an error naming it, rather than sending input the engine rejects.

Keep the on-screen text and captions in the real spelling; only the voice gets the respelling.
Azure's Dragon HD Omni voices ignore `<phoneme>`: for them render `text` (or give both files and say which is which).

## 4. Check every take

```bash
vc check take.wav script.txt --lexicon lexicon.yaml          # the script as written, not the respelled text
vc check --batch takes.json --lexicon lexicon.yaml          # many takes, models loaded once (much faster)
```

`takes.json` is a list of `{"audio": "line01.wav", "script": "The line as written."}` (or a text file of
`audio<TAB>script` lines; paths relative to it). Check every line of a narration in one batch rather than one call
per take: loading the models takes longer than checking a short take. When you'll record, check and re-record in
a loop, start `vc serve` in the background once (it keeps the models loaded on 127.0.0.1:8767); every later
`vc check` uses it automatically, about 3 s for a short take instead of 8. Each report has `timing` (seconds per
step). A take can be any length (a whole chapter is heard in overlapping 30-second windows), but short takes are
cheaper to re-record: check line by line when the pipeline records line by line.

The report's `passed` is true only if all of these hold, and `reasons` says which failed:

- **match**: Whisper's transcript against the script, on letters (numbers compared in whichever reading fits).
- **slips**: words the letter model (no language model, so it can't "correct" anything) can't find in the audio,
  or that it scores low where Whisper also disagrees. Catches dropped small words ("zero [to] two hundred"),
  substitutions Whisper smooths over ("confident and wrong" for "confident in wrong"), truncated takes.
- **names**: each lexicon word with `ipa`, cut from the take and transcribed into phones; `heard`, `target`,
  `score` and `diff` (= same, ~ close, ! different, - missing) show exactly what was said. Accent-level
  differences (a dropped r, cot/caught, an American flapped d, unstressed vowel colour) don't fail it; a wrong
  consonant, a wrong stressed vowel or a missing sound does.
- **pace** (when `pace_syl_s` is set, or `--pace LOW HIGH`) and **tone** (when `flat` limits are set): syllables per
  second of speech, and pitch range / stressed word / sung ending against the sentence's own drift.

When a take fails:

- **A name**: read `heard` against `target`. If the voice consistently says it wrong from the respelling, try the
  next `respell` candidate, or switch to an IPA-capable format if the engine has one, and re-record that line.
  If `heard` looks right to you and only an accent detail differs, explain that rather than loop.
- **Slips or match**: re-record the line (TTS takes vary; a new seed usually fixes it). Repeated failure on the
  same word means the text is hard to read aloud: respell it or reword the sentence, with the user's agreement.
- **Pace or tone**: adjust the engine's speed or style instructions, then re-record.

Give each line a few attempts (three to five), then stop and report what still fails, with the `heard` phones and
`diff`, so a person can listen. Never claim a take passed without running `check` on it.

## 5. The reverse: proofread a transcript against its audio

When the audio is the truth and the text is in doubt (a Whisper transcript, captions, meeting notes, a typed
manuscript of a recording):

```bash
vc proof talk.mp3 talk.srt --lexicon names.yaml --write talk.fixed.srt   # .txt .md .srt .vtt, Whisper .json, or the text
vc proof talk.mp3 talk.txt --second other_tool.txt                       # another tool's transcript as the second opinion
```

It reports `issues`, each with `kind`, `at` (seconds), `line`/`cue`, `transcript` (what is written), `suggest`,
`confidence`, `why` and `evidence`:

- **unheard**: text that isn't in the audio (a hallucinated "Thanks for watching", a line from another take).
- **missing**: speech the transcript skips.
- **misheard**: a second transcript (Whisper by default) wrote another word and the letter model, which has no
  language model, finds that word in the audio and not the transcript's; or the letter model alone clearly hears
  something else.
- **name**: the audio says a lexicon name (its IPA is searched for in the whole recording, and near-spellings are
  checked by sound) where the transcript wrote something else: "Shivon", "Chevonne" → Siobhan.
  **name-mismatch**: the transcript wrote the name, the audio doesn't sound like it.
- **spelling**: a capitalised word the transcripts spell differently and that isn't in the lexicon. Sound can't
  settle a name's spelling, so it is never auto-fixed: research it (section 2), add it to the lexicon, run again.
- **unsure**: the letter model hears something else, not clearly enough to say what.

So the work is the four moves again, in another order: `vc scan transcript.txt` for the names and terms, research
their spelling and IPA (the attendee list, the show notes and the speakers' own profiles are the best sources for
spelling), put them in a lexicon, then `proof`. `--write` applies only `high`-confidence fixes unless told
`--apply medium` or `all`; a `medium` fix is probably right, and its `why` says what to check (a suggestion
"spelled by ear" is the letter model's rough spelling of what it heard: fix the spelling before taking it). Read
the `low` ones yourself: you know what the sentence means, the ears only know what it sounds like. If `note` says
the transcript looks like another recording's, stop and ask. Details and measurements:
`references/proofreading.md`.

## Report back

Tell the user, briefly: which words you researched and the source for each (and any you couldn't confirm), the
lexicon file you wrote, the render format used, and per line whether it passed, with the reasons for any that
didn't. The lexicon is reusable: next time, `scan` will show those words as already covered. For a proofread
transcript: what was fixed and at what confidence, and each issue left for a person to hear, with its time.

## Limits worth stating

- English scripts and voices (whisper-small.en and the letter model are English-only; the phoneme model is
  multilingual, so foreign names inside English narration are fine).
- The name check needs `ipa` in the lexicon; it's calibrated on 32 takes (8 names, 2 voices) and will mishear
  some rare sounds. Treat a failure as "listen to this", with the diff saying where.
- Rare names are where every model is weakest, which is why the research step matters more than the check.
- `proof` can't tell homophones apart (their/there sound the same): meaning is your job. It was measured on clean
  read speech; on noisy, overlapping or accented speech expect more `low` and `unsure` issues, and trust `high`.

## Files

- `vc.py`: the commands above (`vc <command> --help`). `install.py`: setup. `check.py`: the older one-take form.
- `voicecheck/`: the code. `config.py` lists every setting (thresholds, models); point `VOICECHECK_CONFIG` at a
  YAML file to change them. `examples/openmontage_voice.yaml` is a full real configuration.
- `references/name-research.md`, `references/tts-engines.md`, `references/respelling.md`, `references/checks.md`,
  `references/proofreading.md`.
