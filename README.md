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

---

## Install

**You need:** Windows, macOS or Linux, with Python 3.10–3.13 installed. The installer finds it even if your default
Python is newer. You also need about 4 GB of disk for the models. No GPU is needed: everything runs on the CPU.

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
| GitHub Copilot (CLI, VS Code agent mode, cloud agent) | `gh skill install logitsolutionsai/say-it-right`, or `<project>/.github/skills/say-it-right` |
| Antigravity / Gemini CLI | `<project>/.agents/skills/say-it-right` |
| Anything else that can run shell commands | anywhere; point the agent at `SKILL.md` |

### Step 2: set it up (once)

Ask your agent to *"set up say-it-right"*. It runs the skill's `install.py`, which does three things:
- creates the skill's own virtual environment (`.venv` inside the skill's folder);
- installs the dependencies (on Linux it installs PyTorch's CPU build, which saves about 2 GB);
- runs a self-test.

This takes about a minute with [uv](https://docs.astral.sh/uv/), a few minutes without it. If you skip this step,
the first command that needs the setup tells the agent to run it. You can also run it yourself, from wherever the
skill folder is:

```bash
python ~/.claude/skills/say-it-right/install.py            # set up
python ~/.claude/skills/say-it-right/install.py --models   # ...and download the three models now (~3.5 GB)
python ~/.claude/skills/say-it-right/install.py --check    # is it still set up?
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
Marketplaces → say-it-right → Enable auto-update. If an update replaces the skill's folder, the skill sets itself up
again the next time you use it. That takes about a minute, and the models stay cached.

### Elsewhere

**claude.ai and the Claude desktop app:** zip the folder so that `say-it-right/SKILL.md` is at the top of the zip,
then upload it under Customize → Skills. Researching names and rendering scripts work there. Checking audio needs a
local machine, because PyTorch and the models don't fit in claude.ai's code sandbox.

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

Or run the commands yourself (`vc` = `python vc.py`; any Python works, vc.py runs inside the skill's `.venv`):

```bash
vc scan script.txt                                   # which words need researching
vc lookup "Saoirse Ronan"                            # Wikipedia / Wikidata / Wiktionary / Commons / VOA candidates
vc respell "ˈsɜːrʃə" --name                          # -> Sur-shuh, SUR-shuh, ...
vc render script.txt --lexicon lexicon.yaml -f ssml --voice en-US-AvaNeural   # or text, elevenlabs, cartesia, kokoro, piper
vc export-pls --lexicon lexicon.yaml                 # a PLS dictionary for ElevenLabs / Azure / Polly
vc check take.wav script.txt --lexicon lexicon.yaml  # one JSON report; exit 0 pass, 1 fail, 2 error
vc check --batch takes.json --lexicon lexicon.yaml   # many takes, models loaded once
vc serve                                             # keep the models loaded; later checks use it automatically
```

The lexicon (`examples/lexicon.yaml`) records each word's respelling, IPA and source. Settings (models, thresholds,
pace and tone limits) are in `voicecheck/config.py`; point `VOICECHECK_CONFIG` at a YAML file to change them
(`examples/openmontage_voice.yaml` is a real one). The workflow the agent follows is in [SKILL.md](SKILL.md), with
the research behind it in [references/](references/).

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
python install.py && .venv/Scripts/python -m pip install pytest      # Linux/macOS: .venv/bin/python
python -m pytest                                             # unit tests: no models, a few seconds
VOICECHECK_MODEL_TESTS=1 python -m pytest tests/test_models.py # end to end with the real models (Windows voices)
python benchmarks/speechocean762.py                          # the benchmark below (needs pyarrow)
claude plugin validate .                                     # the plugin manifests
```

| Path | What |
|---|---|
| `SKILL.md`, `references/` | the skill: workflow, name research, TTS engine syntax, respelling key, what the checks measure |
| `vc.py`, `install.py` | the command line; one-step setup |
| `voicecheck/verify.py` | `Checker` (Whisper), `Listener` (letters), `match`, `slips`, `pace`, `prosody`, `emotional` |
| `voicecheck/session.py`, `server.py` | a check of one take with its timings; the warm-model server |
| `voicecheck/text.py`, `ctc.py` | numbers and their readings; CTC forced alignment in numpy |
| `voicecheck/phonemes.py`, `ipa.py` | the phone ear; IPA parsing, comparison and respelling |
| `voicecheck/lexicon.py`, `engines.py` | the lexicon; each TTS engine's syntax, PLS export |
| `voicecheck/lookup.py`, `scan.py` | keyless pronunciation lookup; finding the words that need research |
| `benchmarks/`, `evals/`, `tests/` | the benchmark, the skill's test cases, the unit and end-to-end tests |
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
data from it. Your scripts, recordings and reports stay on your machine: every check runs locally. `vc serve`
listens only on 127.0.0.1, so other computers can't reach it.

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
