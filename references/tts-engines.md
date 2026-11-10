# Forcing one word's pronunciation, engine by engine

Checked against official docs and source code on 2026-11-10. Vendors change these often; when a result surprises
you, re-check the linked page. `vc.py render -f <format>` produces the syntax in the "format" column.

## Contents
- [Hosted engines](#hosted-engines)
- [Open-source engines](#open-source-engines)
- [Picking a format](#picking-a-format)

## Hosted engines

| Engine | Accepts | Format | Caveats | Source |
|---|---|---|---|---|
| ElevenLabs v4 | inline IPA in slashes `/ˈsɛɡɛdi/`; dictionaries (IPA/CMU phoneme rules, alias rules); no SSML | `elevenlabs`, `export-pls` | inline IPA documented for `eleven_v4` only, "results can still vary by voice"; capitals add emphasis | elevenlabs.io/docs/best-practices/prompting/controls |
| ElevenLabs v3, v4 Turbo | dictionaries (phoneme + alias rules); no SSML | `export-pls` | inline slashes not documented for v3 | elevenlabs.io/docs/eleven-agents/customization/voice/pronunciation-dictionary |
| ElevenLabs Flash v2 (English) | SSML `<phoneme>` (cmu-arpabet or ipa), dictionaries | `ssml`, `export-pls` | one word per tag; English only | same as above |
| ElevenLabs Multilingual v2, Flash v2.5, Turbo v2.5 (deprecated) | alias only (dictionary or PLS `<alias>`) | `text`, or `export-pls` with `say` only | phoneme tags and phoneme rules are ignored | elevenlabs.io/docs/models |
| OpenAI gpt-4o-mini-tts, tts-1, tts-1-hd | text only | `text` | **all shut down 2027-01-06**; successor gpt-realtime-2.1-mini is Realtime-API only and takes respellings in its instructions ("Pronounce Kyiv as KEE-iv") | developers.openai.com/api/docs/deprecations |
| Azure AI Speech (neural) | SSML `<phoneme>` (ipa, sapi, ups, x-sampa), `<sub alias>`, `<lexicon uri>` (PLS) | `ssml`, `export-pls` | unknown phones return HTTP 400; lexicon ≤ 100 KB, one locale, cached 15 min; IPA in a lexicon can't contain spaces | learn.microsoft.com/azure/ai-services/speech-service/speech-synthesis-markup-pronunciation |
| Azure HD voices | DragonHD: `<phoneme>` yes; Dragon HD Omni: **no `<phoneme>`**; both: `<sub>`, alias-only lexicons | `ssml` (DragonHD) or `text` | `enhancePronunciation=true` is best-effort | learn.microsoft.com/azure/ai-services/speech-service/high-definition-voices |
| Google Cloud TTS (Standard, WaveNet, Neural2, Studio) | SSML `<phoneme>` (ipa, x-sampa), `<sub>`, `customPronunciations` in the request | `ssml` | one word per `<phoneme>`; customPronunciations must match the phrase exactly | docs.cloud.google.com/text-to-speech/docs/ssml |
| Google Chirp 3 HD | SSML `<phoneme>`/`<sub>` **synchronous requests only**; `custom_pronunciations` (IPA, X-SAMPA) incl. streaming | `ssml` | no SSML when streaming; custom pronunciations unavailable in 27 locales (hu-hu, cs-cz, ...) | docs.cloud.google.com/text-to-speech/docs/chirp3-hd |
| Amazon Polly standard / neural / long-form | SSML `<phoneme>` (ipa, x-sampa), `<sub>`, PLS lexicons via PutLexicon + `LexiconNames` | `ssml`, `export-pls` | ≤ 5 lexicons per request; lexicon language must match the voice; SSML `<lexicon>` tag unsupported | docs.aws.amazon.com/polly/latest/dg/phoneme-tag.html |
| Amazon Polly generative | `<phoneme>` "partial availability", `<sub>` yes | test first | which phonemes work is undocumented | docs.aws.amazon.com/polly/latest/dg/supportedtags.html |
| Cartesia Sonic 3 / 3.5 / 3.6 | inline `<<k\|ɑ\|ɹ\|ˈ\|t\|i\|ʒ\|ə>>`; dictionaries (IPA or sounds-like) | `cartesia` | one symbol per `\|`; **every word needs a primary stress**; no length marks; English phone table only | docs.cartesia.ai/build-with-cartesia/capability-guides/phonemes |

## Open-source engines

| Engine | Accepts | Format | Caveats | Source |
|---|---|---|---|---|
| Kokoro (misaki G2P) | `[word](/phonemes/)` in misaki's symbols | `kokoro` | English only; stress mark right before the vowel; misaki symbols (A=eɪ, I=aɪ, W=aʊ, Y=ɔɪ, O=oʊ, ʤ, ʧ); **unknown symbols are silently dropped**, so `vc.py` maps foreign sounds to the nearest English one | github.com/hexgrad/misaki |
| Piper (OHF-Voice/piper1-gpl) | `[[ espeak phonemes ]]` | `piper` | must match espeak-ng's IPA for the voice's language; `espeak-ng -v en-us --ipa=3 -q word` shows the house style | github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md |
| Coqui XTTS-v2 | text only | `text` | the text cleaner lower-cases everything: capitalised stress does nothing | github.com/idiap/coqui-ai-TTS |
| Resemble Chatterbox | text only | `text` | retired research, unmaintained since Aug 2026 | github.com/resemble-ai/chatterbox |
| Qwen3-TTS | text only | `text` | `instruct` steers tone, not pronunciation | github.com/QwenLM/Qwen3-TTS |
| Kyutai Pocket TTS | text only | `text` | SentencePiece tokenizer | github.com/kyutai-labs/pocket-tts |
| Gradium Phonon | text only | `text` | Gradium's cloud API has dictionaries, not tied to Phonon | github.com/gradium-ai/xn-ptts |
| Sesame CSM-1B, Dia / Dia2 | text only | `text` | weak outside English | github.com/SesameAILabs/csm, github.com/nari-labs/dia |

## SSML details `vc.py render -f ssml` takes care of

- Azure requires `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">` and a
  `<voice name="...">`: pass `--voice en-US-AvaNeural` (or the user's voice). Google and Polly take the same document
  or a bare `<speak>` (`--bare`).
- For English voices the `<phoneme>` IPA is converted to the US-English phone set (no length marks, ɹ, ɝ/ɚ, no
  foreign sounds), because Azure rejects phones outside the locale's set with HTTP 400. The lexicon keeps the
  dictionary IPA; only the engine input changes. Dragon HD Omni voices ignore `<phoneme>`: use `text` for them.
- A possessive carries its sound in the IPA (`Nguyen's` → `wɪnz`): a `<phoneme>` around "Nguyen's" with the bare
  name's IPA would be read "Win".

## Picking a format

1. If the engine accepts IPA (SSML phoneme, inline IPA, PLS phoneme), use it for every lexicon entry with `ipa`:
   it is more exact than any respelling, especially for foreign sounds.
2. Otherwise use `text`. Respellings are where most of the iteration happens: if `check` reports a name heard
   wrong, try the next candidate from `vc.py respell`, and re-check.
3. For engines with dictionaries (ElevenLabs, Azure, Polly), `export-pls` once and attach the dictionary: the
   on-screen script and captions stay untouched, and every future script benefits.
4. Whatever the format, `check` judges the audio, not the input: it is the only proof the engine obeyed.

Not verified: what Polly generative's "partial" phoneme support covers; whether inline slashes work on ElevenLabs
v3/v4 Turbo; Google customPronunciations locale coverage (the docs contradict each other).
