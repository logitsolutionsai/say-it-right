# say-it-right

**Make AI voice-overs say every word right, names included, and prove it before the audio ships.**

say-it-right is a skill for coding agents (Claude Code, Codex, Cursor, GitHub Copilot...) and a command-line tool.
Given a script, it finds the words a TTS voice will likely get wrong, researches how each name is really said (with a
source), gives the TTS engine that pronunciation in the syntax it understands, and then listens to every recorded
take with three ears to confirm the voice actually said it.

| Ear | Model | Catches |
|---|---|---|
| Transcript | Whisper `openai/whisper-small.en` | wrong or missing words, when Whisper notices |
| Letters | wav2vec2 `facebook/wav2vec2-large-960h-lv60-self`, **no language model** | slips Whisper "corrects" back to the script, dropped small words ("zero [to] two hundred"), truncated takes |
| Phones | wav2vec2 `facebook/wav2vec2-lv-60-espeak-cv-ft` (IPA) | a name said wrong, judged sound by sound against its researched IPA ("Siobhan" read as *yah-bun*) |

Plus pace (syllables per second of speech) and tone (pitch range, stressed word, sung ending).

**And the reverse.** The same three ears proofread a transcript against its audio (`vc proof`): given a recording
and the text some tool wrote of it (Whisper, a meeting notetaker, auto-captions), they find the text that isn't in
the audio, the speech the transcript skips, the words it misheard and the names it misspelled, and write the fixed
transcript back in its own format. See [Proofread a transcript](#proofread-a-transcript-the-reverse).

---

## Install

**You need:** Windows, macOS (Apple Silicon) or Linux, with Python 3.10–3.13. A newer default Python is fine: the
installer finds an older one next to it, or lets [uv](https://docs.astral.sh/uv/) fetch one if uv is installed.
You also need about 5 GB of disk: 3.5 GB for the models and about 1 GB for the packages. No GPU is needed:
everything runs on the CPU.

### Step 1: add the skill to your agent

**Claude Code, as a plugin (recommended).** Type these two commands in Claude Code:

```
/plugin marketplace add logitsolutionsai/say-it-right
/plugin install say-it-right@say-it-right
```

Or do it in one step:
- In Claude Code (2.1.275+): `/plugin install say-it-right --marketplace logitsolutionsai/say-it-right`
- From a terminal (2.1.292+): `claude plugin install say-it-right --marketplace logitsolutionsai/say-it-right`

**Claude Code, as a skill folder.** Clone the repo into one of Claude Code's skill folders:

```bash
# for you, in every project
git clone https://github.com/logitsolutionsai/say-it-right ~/.claude/skills/say-it-right
# the same on Windows (PowerShell)
git clone https://github.com/logitsolutionsai/say-it-right $env:USERPROFILE\.claude\skills\say-it-right
# or for one project only, run from the project's root
git clone https://github.com/logitsolutionsai/say-it-right .claude/skills/say-it-right
```

**Other coding agents.** The skill follows the [Agent Skills](https://agentskills.io/specification) format: `SKILL.md`
at the folder root, and a folder named after the skill. So the same clone works in other agents:

| Agent | Where to clone it |
|---|---|
| OpenAI Codex | `~/.agents/skills/say-it-right` or `<project>/.agents/skills/say-it-right` |
| Cursor | `~/.cursor/skills/say-it-right` or `<project>/.cursor/skills/say-it-right` (Cursor also reads `~/.claude/skills`) |
| GitHub Copilot (CLI, VS Code agent mode, cloud agent) | `~/.copilot/skills/say-it-right` or `<project>/.github/skills/say-it-right` |
| Gemini CLI | `~/.gemini/skills/say-it-right` or `<project>/.agents/skills/say-it-right` |
| Antigravity | `~/.gemini/config/skills/say-it-right` or `<project>/.agents/skills/say-it-right` |
| Anything else that can run shell commands | anywhere; point the agent at `SKILL.md` |

### Step 2: set it up (once)

Ask your agent to *"set up say-it-right"*. It runs the skill's `install.py`, which does three things:
- creates the skill's own virtual environment: `.venv` inside a cloned skill folder, or `~/.say-it-right/venv` for
  the plugin, because Claude Code replaces the plugin's folder at every update (`VOICECHECK_VENV` puts it elsewhere);
- installs the dependencies (on Linux it installs PyTorch's CPU build, which saves about 2 GB);
- runs a self-test.

This takes about a minute with [uv](https://docs.astral.sh/uv/), a few minutes without it. If you skip this step,
the first command that needs the setup tells the agent to run it. You can also run it yourself, from wherever the
skill folder is. On macOS and Linux the command is `python3`, not `python`.

```bash
python ~/.claude/skills/say-it-right/install.py            # set up
python ~/.claude/skills/say-it-right/install.py --models   # ...and download the three models now (~3.5 GB)
python ~/.claude/skills/say-it-right/install.py --check    # is it still set up, and where is its environment?
```

Without `--models`, the models download on the first check of a take. They go into the Hugging Face cache
(`~/.cache/huggingface`), outside the skill's folder, so updating or reinstalling the skill doesn't download them
again.

### Step 3: use it

In Claude Code, type `/say-it-right` to start the skill. If you installed it as a plugin, its name carries the
plugin's prefix: `/say-it-right:say-it-right`. Typing `/say` finds it either way. You don't have to type the
command at all, though. Claude loads the skill by itself whenever you ask for voice-over, TTS, pronunciation or
take-checking work (examples under [Use it](#use-it)). Other agents also load it by its description when a task
matches.

### Update or remove

| Installed as | Update | Remove |
|---|---|---|
| Claude Code plugin | `claude plugin update say-it-right@say-it-right`, or `/plugin` → Installed → say-it-right → Update now | `claude plugin uninstall say-it-right@say-it-right` |
| Skill folder | `git -C ~/.claude/skills/say-it-right pull` | delete the folder |

Plugins from third-party marketplaces don't update on their own. To turn automatic updates on, open `/plugin` →
Marketplaces → say-it-right → Enable auto-update. After an update, run `/reload-plugins` or restart Claude Code.
The environment and the models are kept across updates. If an update needs new packages, the next command says so,
and running `install.py` again installs only what changed.

Removing the skill leaves two things on disk; delete them to free the space. The plugin's environment is the folder
`~/.say-it-right` (about 1 GB). The models (about 3.5 GB) are the folders `models--openai--whisper-small.en` and
`models--facebook--wav2vec2-*` under `~/.cache/huggingface/hub`.

### Elsewhere

**claude.ai and the Claude desktop app:** make a zip with `say-it-right/SKILL.md` at its top. In a clone, run
`git archive --format=zip --prefix=say-it-right/ -o say-it-right.zip HEAD`. (GitHub's "Download ZIP" names the
folder `say-it-right-main`, and the upload rejects a folder that isn't named after the skill.) Upload the zip under
Customize → Skills → + → Create skill → Upload a skill. Researching names and rendering scripts work there. Checking
audio needs a local machine, because PyTorch and the models don't fit in claude.ai's code sandbox.

**As a plain Python tool:** `pip install git+https://github.com/logitsolutionsai/say-it-right` gives you a `vc`
command with the same subcommands as `python vc.py`.

---

## Use it

Ask your agent in plain words; the skill triggers on narration, voice-over, TTS, pronunciation and take-checking work:

> "Get this script ready for ElevenLabs v4 so every name is said right, and tell me where each pronunciation came from."
>
> "Did the voice in take_03.wav say *Siobhan* right? If not, what did it say?"
>
> "Check all the takes in out/ against script.txt and re-record the ones that fail."

> "Here's the recording of Tuesday's call and the notetaker's transcript. Fix the transcript: the attendees were
> Siobhan Nguyen and Christian Szegedy."

Or run the commands yourself (`vc` = `python vc.py`; any Python works, vc.py runs inside the skill's own
environment):

```bash
vc scan script.txt                                   # which words need researching
vc lookup "Saoirse Ronan"                            # Wikipedia / Wikidata / Wiktionary / Commons / VOA candidates
vc respell "ˈsɜːrʃə" --name                          # -> Sur-shuh, SUR-shuh, ...
vc render script.txt --lexicon lexicon.yaml -f ssml --voice en-US-AvaNeural   # or text, elevenlabs, cartesia, kokoro, piper
vc export-pls --lexicon lexicon.yaml                 # a PLS dictionary for ElevenLabs / Azure / Polly
vc check take.wav script.txt --lexicon lexicon.yaml  # one JSON report; exit 0 pass, 1 fail, 2 error
vc check --batch takes.json --lexicon lexicon.yaml   # many takes, models loaded once
vc serve                                             # keep the models loaded; later checks use it automatically
vc proof talk.mp3 talk.srt --lexicon names.yaml --write talk.fixed.srt   # the reverse: fix the transcript
```

The lexicon (`examples/lexicon.yaml`) records each word's respelling, IPA and source. Settings (models, thresholds,
pace and tone limits) are in `voicecheck/config.py`; point `VOICECHECK_CONFIG` at a YAML file to change them
(`examples/openmontage_voice.yaml` is a real one). The workflow the agent follows is in [SKILL.md](SKILL.md), with
the research behind it in [references/](references/).

### Proofread a transcript (the reverse)

`check` trusts the script and judges the audio. `proof` trusts the audio and judges the text:

```bash
vc proof meeting.wav meeting.txt --lexicon names.yaml --write meeting.fixed.txt
```

```text
The transcript a tool wrote                          What proof reports
"Shivon will present the results..."                 name      high    Shivon -> Siobhan   (sounds like the lexicon's ʃɪˈvɔːn)
"...is confident and wrong answers..."               misheard  medium  and -> in           (Whisper wrote "in"; the letters fit it)
"...goes from zero two hundred quickly."             missing   high    insert "to"
"Christian Segedi wrote the paper,"                  name      high    Segedi -> Szegedy
"and Chevonne checked it twice."                     name      medium  Chevonne -> Siobhan (found by sound alone)
"Thanks for watching, and don't forget to..."        unheard   high    delete: nothing is said there
```

That is a real run (a 17-second recording, six errors planted in its transcript, none missed, nothing else
flagged). `--write` applied the four `high` fixes; `--apply medium` applies all six. The swap of two short words
is held at `medium` on purpose: a/the, in/of and and/in blur in fluent speech, and on real recordings that is
where a wrong fix would come from. How it decides:

- **Text that isn't in the audio, and speech that isn't in the text.** The transcript is aligned to the audio by
  the letter model. Words it can't find are `unheard`; speech it hears between or around the transcript's words
  is `missing`.
- **Two transcripts and a referee.** Whisper transcribes the audio again (or pass another tool's transcript with
  `--second`). Where the two disagree, both wordings are aligned to the same stretch of audio by the letter model,
  which has no language model and so no view on what would make a good sentence. The better fit wins; if the
  original fits better, nothing is reported.
- **Names by sound.** A name's spelling doesn't follow its sound, so letters can't settle it: "Shivon" fits the
  audio of *Siobhan* better than "Siobhan" does. Names go in the lexicon with their IPA, and proof searches the
  phones of the whole recording for each one. Where the audio says the name and the transcript wrote anything
  else, the fix is the lexicon's spelling. A disagreement about a name that isn't in the lexicon is reported as
  `spelling` and never auto-fixed.

Every issue carries its time, line or caption number, evidence and a confidence. `--write` applies the `high`
ones (`--apply medium` or `all` for more) and keeps the format: SRT and WebVTT keep their timing lines and tags,
Whisper JSON keeps its segments, plain text keeps its punctuation. Any length of recording works: an hour of audio
is heard in overlapping 30-second windows and aligned in sections, in memory that doesn't grow with the length.
Details: [references/proofreading.md](references/proofreading.md); measurements under
[Results](#proofreading-a-transcript).

What it is for: podcast and meeting transcripts that misspell guests and attendees, captions and subtitles before
they are published, a narrator's recording against the manuscript, dictation where a term matters, a transcript a
model may have padded with text nobody said, and cleaning speech datasets (keep the utterance, fix the word).

---

## Results

Measured 2026-11-10 on a laptop (Intel i7-1280P, 16 GB), CPU only, Windows 11, Python 3.13, torch 2.11.
"Before" is the checker as it was split out of OpenMontage, run in OpenMontage's environment on the same audio
with the same script.

### Same takes, before and after

Real recordings from the Windows voices (David, Zira). Truth comes from what each voice was given to read (the
slips were made by giving it the wrong words) and, for names, from the phones it actually produced.

| Take | What the voice did | Before | After |
|---|---|---|---|
| "In 1959 the network had 784 weights." | read it right | ❌ fail: 4 false slips (*in, one, thousand, hundred*) | ✅ pass |
| "Doctor Nguyen agreed." | read it right (Whisper writes "Dr.") | ❌ fail: match 0.67 | ✅ pass |
| "Ask Siobhan about it." | said *Shiv-awn* (right) | ❌ fail: slip *siobhan* | ✅ pass |
| "Ask Siobhan about it." | said *See-yah-bun* (wrong) | ❌ fail, same as the right read | ❌ fail: heard `j ɑ b ə n`, want `ʃ ɪ v ɔ n` |
| "Christian Szegedy wrote the paper." | said *Seg-eh-dee* (right) | ❌ fail: slip *szegedy* | ✅ pass |
| "The model is confident in wrong." | said "and" for "in" | ❌ fail | ❌ fail: slip *in* |
| "...goes from zero to two hundred quickly." | dropped "to" | ❌ fail | ❌ fail: slip *to* |

**All 32 name takes.** Each of 8 hard names was read in 2 voices, once from a researched respelling and once from
the raw spelling. Of those reads, 22 were right and 10 wrong; one voice read *Sergey* wrong inside "Sergey Ioffe".

| | Before | After |
|---|---|---|
| Right reads that pass | 8 / 22 | **22 / 22** |
| Wrong reads that fail | 10 / 10 | **10 / 10** |

Before, a name passed only if Whisper happened to spell it and the letter model could read it off the spelling
(Joaquin, Sergey). Every other name failed, whether it was read right or wrong. After, each researched name is
judged by its sounds against its IPA. These 32 takes are also where the name thresholds were set, so treat them
as a fit, not a held-out test. The benchmark below is held out.

### Benchmark: spotting a mispronounced word (speechocean762)

There is no public benchmark of TTS name pronunciation. The closest held-out test of the core step (did this word
sound like its target phones?) is [speechocean762](https://huggingface.co/datasets/mispeech/speechocean762): 2,500
test utterances (15,967 words) read by non-native English speakers, half of them children, with every word and phone
scored by five experts. `benchmarks/speechocean762.py` runs the checker's ears on each word, exactly as `vc check`
does for a lexicon word, with no training on the dataset.

| Method | PCC with experts' word accuracy | ROC-AUC: word has an incorrect phone | PCC, words of 2+ syllables | ROC-AUC, 2+ syllables |
|---|---|---|---|---|
| Plain phone error rate (baseline) | 0.287 | 0.788 | 0.462 | 0.838 |
| Letter ear alone | 0.399 | 0.845 | 0.509 | 0.836 |
| **Phone ear: feature-weighted, accent-tolerant distance (shipped)** | **0.379** | **0.843** | **0.535** | **0.858** |
| Both ears averaged | 0.442 | 0.872 | 0.571 | 0.873 |
| *For scale: GOPT (supervised, trained on speechocean762's training split)* | *≈0.53* | | | |

- The shipped distance beats plain phone error rate on every subset: +0.09 PCC and +0.055 AUC on all words. It
  does as well for adults (AUC 0.845 vs 0.782) as for children (0.844 vs 0.796).
- On all words it is level with the letter ear. On longer words it is ahead (PCC 0.535 vs 0.509 on 2+ syllables),
  and that is where names live.
- Averaging the two ears would do better again (0.442 / 0.872). The shipped checker can't use that on names,
  because the letter ear needs a spelling that matches the sound. That's the reason a name needs a phone check at
  all.
- The pass/fail rule is strict by design: one clear wrong sound fails a word. On learners' speech it flags 27% of
  words and catches 78% of the words experts marked wrong (precision 21%). A false alarm costs one re-take; a
  misread name that ships costs more. TTS voices make far fewer accent-level deviations than learners do.

Full numbers: `benchmarks/results/speechocean762_summary.json`.

### The skill, with and without

Three test tasks (`evals/`), each run by Claude Code with the skill and without it (same model, same files, one run
each).

| Task | With the skill | Without |
|---|---|---|
| Research 5 names for a plain-text TTS, with sources | 4 / 4 checks, 603 s | 2 / 4, 484 s |
| Did this take say *Siobhan* right? (it didn't) | 3 / 3, **280 s, 12 tool calls** | 2 / 3, 1,443 s, 38 tool calls |
| SSML with numbers and acronyms | 3 / 3, 492 s | 2 / 3, 221 s |
| **All** | **10 / 10** | **6 / 10** |

Three of the ten checks look for the skill's own artefacts, such as a lexicon file with IPA, so a run without the
skill can't pass them. On the 7 checks that look only at the outcome, the score was 7/7 with the skill and 6/7
without; the miss was an unsourced Nguyen pronunciation. The largest difference was judging audio. Without the
skill, the agent reached the same verdict on the misread take only after building its own phoneme pipeline: five
times the time and three times the tool calls.

### Proofreading a transcript

Measured 2026-11-11 on the same laptop. There is no public benchmark for correcting someone else's transcript
against its audio, so `benchmarks/proof_librispeech.py` makes one from
[LibriSpeech test-clean](https://www.openslr.org/12) (real people reading, exact reference text): 150 utterances
of 10–20 s (33.6 minutes, 5,513 words), each turned into a "transcript from some tool" with one known error, plus 25
left untouched to count false alarms.

| Injected error (25 of each) | Found, with Whisper as second opinion | Fixed by `--write` (high only) | Found, letter ear alone | Fixed, letter ear alone (`--apply medium`) |
|---|---|---|---|---|
| A word swapped for an unrelated one | 100% | 84% | 100% | 28% |
| A word swapped for a similar one (walked → walks) | 100% | 16% (68% with `--apply medium`) | 96% | 52% |
| One or two words dropped | 100% | 92% | 96% | 32% |
| One or two words added | 100% | 88% | 100% | 84% |
| A sentence nobody said, appended | 100% | 96% | 100% | 80% |
| **All 125** | **125 / 125** | | **123 / 125** | |

| | With Whisper | Letter ear alone |
|---|---|---|
| Word error rate of the transcripts, before | 4.39% | 4.39% |
| ...after `--write` (high-confidence fixes) | **0.65%** | 2.54% |
| ...after `--write --apply medium` | 0.83% | 1.41% |
| Wrong high-confidence fixes in 5,513 words | 1 | 0 |
| Untouched transcripts with nothing reported at medium or high | 21 / 25 | 25 / 25 |

- With a second opinion, every injected error was found, and applying only the `high` fixes removed 85% of the
  word errors while making one wrong change ("a" → "the") in 5,513 words.
- Similar-sounding swaps are found but mostly held at `medium`: by design, since sound alone can't separate
  near-homophones and the report says to check the meaning.
- Four untouched transcripts drew a `medium` issue: two were Whisper's modern spelling against the book's
  ("removed" for "remov'd", "watery" for "watry"), two were small words Whisper heard differently ("condensed" for
  "condense", "a" for "the"). Medium fixes like these are why `--apply medium` scores slightly worse than `high`
  against these references, and why `high` is the default.
- The letter ear alone (no second transcript: the case where the transcript *is* Whisper's) finds nearly as much
  but can only offer what it heard, spelled by ear, so its fixes are `medium` and need a reader.

Read these for what they are. The errors are injected, not made by a real recogniser. The letter model was trained
on LibriSpeech's training sets, so this is its home ground: noisy, conversational or accented speech will score
lower. The one threshold tuned on this data (`proof_heard_gain`) was set on 60 other utterances, left out of the
150. Names aren't in this benchmark (LibriSpeech has no pronunciation glossary); for those there are the planted
runs, made with the Windows voices: the 17-second recording under
[Use it](#proofread-a-transcript-the-reverse) (all three misspelled names found; two fixed at `high`, and the one
found by sound alone at `medium`), a shorter version of which is an end-to-end test (`tests/test_models.py`); and
a 130-second narration given as captions with seven errors planted (two swapped words, a wrong year, a wrong
decimal, a deleted sentence, a hallucinated line, a British spelling): 29 word errors before, 1 after
`--apply medium` (the British spelling, which isn't an error), nothing else flagged.
Full numbers: `benchmarks/results/proof_librispeech_summary.json`.

### Latency

Seconds of wall time to check one take, including starting `vc check`. "Before" is the faster of two runs:

| Take (audio length) | Before: one process per take | After: one process per take | After, `vc serve` running | ... + int8 Whisper |
|---|---|---|---|---|
| "The network learns one weight at a time." (2.8 s) | 11.0 | 7.6 | 2.8 | 2.4 |
| "In 1959 the network had 784 weights." (4.9 s) | 12.2 | 8.1 | 3.3 | 2.8 |
| "...from zero to two hundred quickly." (3.5 s) | 11.4 | 8.0 | 2.9 | 2.4 |
| "Ask Siobhan about it." (2.0 s, + phone check) | 11.0 | 9.1 | 3.7 | 3.3 |
| "The model is confident in wrong." (2.6 s, a failing take) | 11.0 | 11.0 | 6.3 | 4.9 |
| a 117-word paragraph (40.3 s) | 48.5 | 25.5 | 21.5 | 19.4 |

What changed:

- **Whisper decodes greedily first.** Beam search (5 beams, OpenAI's default) runs only when the greedy transcript
  doesn't match the script, and then it decides. Passing takes get transcribed about twice as fast. Failing takes
  pay for both passes, which is why the failing take above didn't get faster in a single process. On the 38 test
  takes, every verdict was the same as always using 5 beams.
- **Models load from the local cache, without asking Hugging Face whether they changed.** That check cost about
  8 s on every start (14 s to load instead of 4.5 s, measured on a fresh plugin install). Only a model that isn't
  cached yet is downloaded, and checks also work offline.
- **Models load once.** `vc check --batch` checks many takes per process. `vc serve` keeps the models loaded, and
  every later `vc check` uses it automatically. Loading costs about 4 s plus 1 s to start Python. The very first
  load takes up to two minutes, while 3.5 GB of models are read from disk.
- **Pitch tracking runs only when asked for**, when tone limits are set or with `--tone`. It costs 0.7 s on a
  short take and 1.5 s on the 40 s one.
- **int8 Whisper** (`quantize: [checker]`) saves another 0.4–2 s per take with the server running. Loading costs
  about 2.5 s more, so it doesn't pay off one process at a time. It changed no verdict on the 38 test takes. It is
  off by default because the transcript changes slightly.
- **int8 for the other two models was rejected.** `quantize: true` makes short takes about a third faster, but it
  moved a correct *Siobhan* below the name threshold (37/38).

Where the time goes now, for a short take with the server running: Whisper about 2 s (1.35 s of that is its
encoder, which always reads a 30-second window), the letter ear 0.5–1 s, and the phone ear about 1 s per take with
names. Run `python benchmarks/latency.py takes.json` to measure your own machine.

---

## Development

```bash
python install.py                                            # the environment: .venv in this folder
uv pip install --python .venv/Scripts/python.exe pytest      # without uv: .venv/Scripts/python -m pip install pytest
.venv/Scripts/python -m pytest                               # unit tests: no models, a few seconds
VOICECHECK_MODEL_TESTS=1 .venv/Scripts/python -m pytest tests/test_models.py   # the real models (Windows voices)
.venv/Scripts/python benchmarks/speechocean762.py            # the name benchmark under Results (needs pyarrow)
.venv/Scripts/python benchmarks/proof_librispeech.py --data <LibriSpeech/test-clean>   # the proofreading one
claude plugin validate .                                     # the plugin manifests; it doesn't read SKILL.md
skills-ref validate .                                        # SKILL.md against the Agent Skills specification
```

On macOS and Linux, `python` is `python3` and `.venv/Scripts/python` is `.venv/bin/python`. An environment made by
uv has no pip of its own, which is why pytest is installed with `uv pip`.
[skills-ref](https://github.com/agentskills/agentskills/tree/main/skills-ref) wants the folder to be named
`say-it-right`, like the skill.

| Path | What |
|---|---|
| `SKILL.md`, `references/` | the skill: workflow, name research, TTS engine syntax, respelling key, what `check` and `proof` measure |
| `vc.py`, `install.py` | the command line; one-step setup |
| `voicecheck/verify.py` | `Checker` (Whisper), `Listener` (letters), `match`, `slips`, `pace`, `prosody`, `emotional` |
| `voicecheck/session.py`, `server.py` | a check of one take with its timings; the warm-model server |
| `voicecheck/proof.py` | `vc proof`: transcript formats, the issues, the fixes |
| `voicecheck/text.py`, `ctc.py` | numbers and their readings; CTC forced alignment, free decoding and section cuts in numpy |
| `voicecheck/phonemes.py`, `ipa.py` | the phone ear; IPA parsing, comparison, spotting a name in a recording, respelling |
| `voicecheck/lexicon.py`, `engines.py` | the lexicon; each TTS engine's syntax, PLS export |
| `voicecheck/lookup.py`, `scan.py` | keyless pronunciation lookup; finding the words that need research |
| `benchmarks/`, `evals/`, `tests/` | the benchmarks, the skill's test cases, the unit and end-to-end tests |
| `reference/` | OpenMontage's audits built on the checker (they import OpenMontage and run there) |
| `.claude-plugin/` | Claude Code plugin and marketplace manifests |

**Windows note.** Smart App Control can block a freshly released wheel's DLL ("DLL load failed ... blocked by an
Application Control policy"): on 2026-11-10 torch 2.15.0, numba 0.68.1 and pyarrow 22 were blocked for a few hours
after release. `install.py` falls back to `requirements-lock.txt`, the exact set this was tested with.

**OpenMontage.** This started as the take checker of logitsolutionsai's OpenMontage pipeline (split out 2026-11-10).
`voicecheck.verify` keeps the names and signatures OpenMontage's `narrate.py` calls, so it can switch to
`pip install -e <this folder>`. Behaviour changes since the split: numbers and abbreviations ("Dr.") no longer
fail a correct read, Whisper decodes greedily with a 5-beam retry on a mismatch, an unalignable take fails instead
of passing, `slips` scores the words the listener actually aligned, logs go to stderr,
the pitch range adapts to high voices (`PROSODY_V` 4), and torchaudio is gone. Bugs fixed here in `reference/`
(OpenMontage's copies still have them) are listed in the commit history.

## Privacy

say-it-right collects nothing. It has no account, no API key and no telemetry, and logitsolutionsai receives no
data from it. Your scripts, recordings, transcripts and reports stay on your machine: every check and every
proofread runs locally. `vc serve` listens only on 127.0.0.1, so other computers can't reach it, and it turns away
requests that come from a web page.

It contacts other services only in these cases:

| When | What is sent | To |
|---|---|---|
| Setup (`install.py`) | ordinary package downloads | PyPI, and download.pytorch.org on Linux |
| The first time the models are needed (a check, `vc scan`, or `install.py --models`) | model downloads, once; after that they load from the local cache | Hugging Face |
| Researching a name (`vc lookup`) | the name being looked up | Wikipedia, Wikidata, Wiktionary, Wikimedia Commons, VOA Pronounce |
| Turning a reference recording into phones (`vc hear <url>`) | a download request for that URL | the site in the URL, usually upload.wikimedia.org |

Lookup requests carry a User-Agent that names the project and its contact (the repository URL and
logitsolutionsai@gmail.com), as Wikimedia's policy for automated tools asks. Set `VOICECHECK_CONTACT` to use your
own contact instead. When your agent searches the web for how someone says their name, the search goes through
the agent's own web tools, under that agent's privacy terms. Each service above has its own privacy policy.

## License

[MIT](LICENSE). The models it downloads have their own licences: Whisper (MIT), the two wav2vec2 models
(Apache-2.0). Pronunciation data `lookup` fetches carries its source's licence (Wikipedia and Wiktionary text
CC BY-SA, Commons recordings as listed per file); the benchmark data (speechocean762) is Apache-2.0.
