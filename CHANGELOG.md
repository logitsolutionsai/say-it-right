# Changelog

## 0.3.0 (2026-11-11)

### New: proofread a transcript against its audio

- `vc proof AUDIO TRANSCRIPT [--lexicon FILE] [--second FILE | --no-second] [--write FILE] [--apply high|medium|all]`:
  the checker run in reverse. It reports text that isn't in the audio (`unheard`), speech the transcript skips
  (`missing`), words a second transcript and the letter model both contradict (`misheard`), lexicon names the audio
  says but the transcript spelled another way (`name`), and names it can't settle (`spelling`), each with its time,
  line or caption number, evidence and a confidence. `--write` applies the fixes in the transcript's own format
  (plain text, Markdown, SRT, WebVTT, Whisper JSON). See `references/proofreading.md`.
- `ipa.spot`: find every place a name's IPA is said in a whole recording.
- `vc serve` answers `proof` too.
- Measured on 150 LibriSpeech utterances with injected errors: all 125 found with Whisper as the second opinion,
  word error rate 4.39% to 0.65% with the high-confidence fixes, one wrong high-confidence fix in 5,513 words
  (`benchmarks/proof_librispeech.py`; limits in the README).

### Long recordings

- A take or recording of any length now works. The two wav2vec2 models hear audio over 40 seconds in overlapping
  30-second windows, and the text is aligned in sections cut where the text and the model's own hearing agree, so
  memory no longer grows with the length (heard whole, a five-minute take needed more than 10 GB). Section-wise
  alignment gives the same word scores as whole alignment on the same posteriors.

### Fixed: correct takes that failed, and wrong takes that passed

- Currency (`$5.00`, `$0.99`, `$.50`, `$1,234.50`), `$20 T-shirt` read as "twenty trillion", `$5 Million`, mixed
  fractions (`2½`, `3 1/2`), `#1` and `No. 5`, degrees (`5°C`), Roman numerals (`World War II`), `1000s of`,
  `2.0`, and times written `10.30am` now have the readings a voice gives them.
- British and American spellings (colour/color, centre/center, -ise/-ize), OK/okay and similar pairs count as the
  same word in the transcript comparison.
- A lexicon name no longer hides every ordinary word spelled like it or like its respelling ("We win when Win
  wins"), and the name check listens at the name, not at an earlier ordinary word.
- A number swapped for another number (15 for 50) is a slip whatever its acoustic score. Before a take fails on
  a word Whisper wrote differently, Whisper's 5-beam decoder now gets the last word on it, as it already did for
  a low transcript match.
- Lexicon words next to a hyphen (`Szegedy-style`, `pre-AI`) and plural acronyms (`GPUs`) are matched.
- IPA with tie bars (`t͡ʃ`), variants (`ˈniːtʃə, -tʃi`), tone letters, syllabic consonants and `ç` is read
  correctly; respellings contain only letters, hyphens and spaces.

### Fixed: engine output

- Azure SSML marks every syllable, so the stress lands on the right one; phones Azure rejects are mapped to the
  nearest English sound or stop the render with an error naming them.
- Cartesia output uses only Sonic 3's symbols; Kokoro and Piper output only theirs.
- Multi-word lexicon entries become one `<phoneme>` per word (and one pair of slashes per word for ElevenLabs);
  PLS export follows.
- The right occurrence is tagged when the same letters appear earlier in the text ("The MAIN idea of AI").
- `references/tts-engines.md` is re-checked against each engine's documentation, with eleven more engines.

### Fixed: command line and setup

- Bad input (broken YAML, a lexicon value of the wrong type, a batch entry without text, an empty batch, a script
  file that isn't there) exits 2 with a one-line error; an empty batch no longer reports `passed`.
- The plugin's environment lives in `~/.say-it-right/venv` and survives plugin updates (a cloned skill folder
  keeps its `.venv`); `VOICECHECK_VENV` puts it anywhere. With uv installed, `install.py` no longer needs a
  suitable Python on the machine. A half-made environment is rebuilt instead of failing forever.
- `vc serve` refuses requests that don't come from this machine's own tools (a web page can't reach it), and a
  client ignores a server left running from another version.

## 0.2.0 (2026-11-10)

First public release: scan, lookup, respell, render, export-pls, check, serve.
