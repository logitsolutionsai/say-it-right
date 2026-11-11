# Forcing one word's pronunciation, engine by engine

Checked against official docs and source code on 2026-11-11. Vendors change these often; when a result surprises
you, re-check the linked page. `vc.py render -f <format>` produces the syntax in the "format" column.

## Contents
- [Hosted engines](#hosted-engines)
- [Open-source engines](#open-source-engines)
- [SSML details](#ssml-details-vcpy-render--f-ssml-takes-care-of)
- [PLS details](#pls-details-vcpy-export-pls-takes-care-of)
- [Other engines](#other-engines-no-format-of-their-own-yet)
- [Picking a format](#picking-a-format)

## Hosted engines

| Engine | Accepts | Format | Caveats | Source |
|---|---|---|---|---|
| ElevenLabs v4 | inline IPA in slashes `/ˈsɛɡɛdi/`; dictionaries (IPA/CMU phoneme rules, alias rules); no SSML | `elevenlabs`, `export-pls` | inline IPA documented for `eleven_v4` only, "results can still vary by voice"; every example in the docs has the slashes inside double quotes (`"/ˌbaɪoʊˈkemɪstri/"`) and it is unclear whether they are required: `vc.py` writes bare slashes, so if a take reads the symbols aloud add the quotes; capitals add emphasis | elevenlabs.io/docs/best-practices/prompting/controls |
| ElevenLabs v3, v4 Turbo | dictionaries (phoneme + alias rules); no SSML | `export-pls` | inline slashes not documented for these | elevenlabs.io/docs/eleven-agents/customization/voice/pronunciation-dictionary |
| ElevenLabs Flash v2 (English) | inline `<phoneme>` tags (cmu-arpabet or ipa), dictionaries | `export-pls` | **not `-f ssml`**: ElevenLabs reads `<phoneme>` and `<break>` only, not `<speak>`, `<voice>` or `<sub>`; a hand-written tag covers one word; CMU Arpabet is "more reliable" than IPA here; English only | same page; elevenlabs.io/blog/ssml-tags-explained |
| ElevenLabs Multilingual v2, Flash v2.5, v3 Conversational; Turbo v2.5 and Turbo v2 (deprecated) | alias only (dictionary or PLS `<alias>`) | `text`, or `export-pls` with `say` only | phoneme rules are skipped ("other models... use the default pronunciation") | elevenlabs.io/docs/models |
| OpenAI gpt-4o-mini-tts, tts-1, tts-1-hd | text only | `text` | **all shut down 2027-01-06**; successor gpt-realtime-2.1-mini is Realtime-API only and takes respellings in its instructions ("Pronounce Kyiv as KEE-iv") | developers.openai.com/api/docs/deprecations |
| Azure AI Speech (neural) | SSML `<phoneme>` (ipa, sapi, ups, x-sampa), `<sub alias>`, `<lexicon uri>` (PLS) | `ssml`, `export-pls` | unknown phones return HTTP 400; a stress mark stresses the next syllable only if every syllable is marked with `.`; lexicon ≤ 100 KB, one locale, cached 15 min, no phoneme for a phrase, no spaces in its IPA | learn.microsoft.com/azure/ai-services/speech-service/speech-synthesis-markup-pronunciation |
| Azure HD voices | DragonHD: `<phoneme>` yes; Dragon HD Omni: **no `<phoneme>`**; both: `<sub>`, alias-only lexicons | `ssml` (DragonHD) or `text` | `enhancePronunciation=true` is best-effort; Dragon HD Flash is not in the docs' SSML table | learn.microsoft.com/azure/ai-services/speech-service/high-definition-voices |
| Google Cloud TTS (Standard, WaveNet, Neural2, Studio) | SSML `<phoneme>` (ipa, x-sampa), `<sub>`, `customPronunciations` in the request | `ssml --bare` | one word per `<phoneme>`; customPronunciations must match the phrase exactly and not sit inside a `<phoneme>`; its en-US phone table differs from Azure's (see SSML details) | docs.cloud.google.com/text-to-speech/docs/ssml, .../docs/phonemes |
| Google Chirp 3 HD | SSML `<phoneme>`/`<sub>` **synchronous requests only** (preview); `custom_pronunciations` (IPA, X-SAMPA) incl. streaming | `ssml --bare` | no SSML when streaming; custom pronunciations unavailable in 27 locales (hu-hu, cs-cz, ...) | docs.cloud.google.com/text-to-speech/docs/chirp3-hd |
| Amazon Polly standard / neural / long-form | SSML `<phoneme>` (ipa, x-sampa), `<sub>`, PLS lexicons via PutLexicon + `LexiconNames` | `ssml --bare`, `export-pls` | no `<voice>` or `<lexicon>` tag (an unsupported tag is an error); ≤ 5 lexicons per request; lexicon language must match the voice | docs.aws.amazon.com/polly/latest/dg/supportedtags.html, .../phoneme-tag.html |
| Amazon Polly generative | `<phoneme>` "partial availability", `<sub>` yes | test first | which phonemes work is undocumented | docs.aws.amazon.com/polly/latest/dg/supportedtags.html |
| Cartesia Sonic 3 / 3.5 / 3.6 | inline `<<k\|ɑ\|ɹ\|ˈ\|t\|i\|ʒ\|ə>>`; dictionaries (IPA or sounds-like, `pronunciation_dict_id`) | `cartesia` | one symbol per `\|`; only its 41 symbols, anything else is invalid: no `ʌ` (write `ə`), no `ɚ` or `ə\|ɹ` (write `ɝ`), `ɡ` and `ɹ`, no length marks; **every word needs a primary stress**; `vc.py` maps to these and stops on a sound it can't map | docs.cartesia.ai/build-with-cartesia/capability-guides/phonemes |

## Open-source engines

| Engine | Accepts | Format | Caveats | Source |
|---|---|---|---|---|
| Kokoro (misaki G2P) | `[word](/phonemes/)` in misaki's symbols | `kokoro` | English only; stress mark right before the vowel; misaki symbols (A=eɪ, I=aɪ, W=aʊ, Y=ɔɪ, O=oʊ, ʤ, ʧ, ɜɹ and əɹ for ɝ and ɚ); **unknown symbols are silently dropped**, so `vc.py` writes only misaki's US-English set: a foreign sound becomes the nearest English one, and one with no English neighbour is an error | github.com/hexgrad/misaki (EN_PHONES.md, misaki/en.py) |
| Piper (OHF-Voice/piper1-gpl) | `[[ espeak phonemes ]]` | `piper` | espeak-phoneme voices only; must match espeak-ng's IPA for the voice's language: `vc.py` writes the en-us style (stress before the vowel, `ɑː ɔː ɜː uː`, `ɚ`), for an English voice; `espeak-ng -v en-us --ipa=3 -q word` shows the house style | github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md |
| Coqui XTTS-v2 | text only | `text` | the text cleaner lower-cases everything: capitalised stress does nothing | github.com/idiap/coqui-ai-TTS |
| Resemble Chatterbox (original, Turbo, Nano, Multilingual) | text only | `text` | "retired research... no longer maintained" since Aug 2026; Turbo's `[laugh]`-style tags are not pronunciation | github.com/resemble-ai/chatterbox |
| Qwen3-TTS | text only | `text` | `instruct` steers tone, not pronunciation | github.com/QwenLM/Qwen3-TTS |
| Kyutai Pocket TTS | text only | `text` | SentencePiece or JSON tokenizer over spelling; 7 languages | github.com/kyutai-labs/pocket-tts |
| Gradium Phonon | text only | `text` | Gradium's cloud API has dictionaries of respellings (`original` → `rewrite`), not IPA | github.com/gradium-ai/xn-ptts, docs.gradium.ai/api-reference/endpoint/create-pronunciation |
| Sesame CSM-1B | text only | `text` | weak outside English | github.com/SesameAILabs/csm |
| Dia / Dia2 | text only | `text` | English only; `[S1]`/`[S2]` and `(laughs)` are speaker and sound cues | github.com/nari-labs/dia, github.com/nari-labs/dia2 |

## SSML details `vc.py render -f ssml` takes care of

- Azure requires `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">` and a
  `<voice name="...">`: pass `--voice en-US-AvaNeural` (or the user's voice). Google and Polly take a bare `<speak>`
  (`--bare`) and no `--voice`: Polly has no `<voice>` tag and rejects tags it doesn't support, and Google's
  `<voice>` wants a Google voice name.
- For English voices the `<phoneme>` IPA is rewritten in US-English phones: no length marks, aspiration, tie bars
  or syllabic marks; ɹ, ɝ/ɚ; a foreign sound becomes the nearest English one (`x` → `k`, `ç` → `h`, a bare `a` →
  `ɑ`, a nasal vowel → vowel + `n`), because Azure rejects phones outside the locale's set with HTTP 400. A symbol
  with no English neighbour (a click) stops the render with an error naming the word, rather than produce a
  document Azure refuses. The lexicon keeps the dictionary IPA; only the engine input changes.
- Every syllable is marked off (`ʃɪ.ˈvɔn`). Azure: "to stress one syllable by placing stress symbol before this
  syllable, you need to mark all syllables for the word. Or else, the syllable before this stress symbol is
  stressed." Polly and Google list `.` as a syllable boundary, so the same string serves all three.
- Where the three phone tables differ the output follows Azure, the only one documented to reject: ASCII `g`
  (Polly and Google list `ɡ` U+0261; Google's own examples use `g`), `ɝ` and short `i u ɑ ɔ` (Google lists only
  `ɚ` and `iː uː ɑː ɔː`), `tʃ dʒ` (Google lists `ʧ ʤ`, Polly `t͡ʃ d͡ʒ`). What Google and Polly do with a phone
  outside their table is not documented: check a take.
- A phrase gets one `<phoneme>` per word (Google: a tag "directs the pronunciation of a single word"). Write a
  phrase's IPA with one word per written word (`Xi Jinping` → `ɕi tɕinpʰiŋ`); otherwise the render stops with an
  error.
- A possessive carries its sound in the IPA (`Nguyen's` → `wɪnz`): a `<phoneme>` around "Nguyen's" with the bare
  name's IPA would be read "Win".
- Dragon HD Omni voices don't support `<phoneme>`: use `text` for them.

## PLS details `vc.py export-pls` takes care of

- `<phoneme>` holds the lexicon's IPA as written (ElevenLabs takes any IPA; Azure's own en-US example has `ː`),
  `<alias>` the respelling of entries without IPA.
- A phrase with IPA is written as one lexeme per word (`Xi Jinping` → `Xi`, `Jinping`): Azure can't give a phrase
  a phoneme and allows no white space in one, and ElevenLabs applies phonemes word by word. The limit: each word
  is then said that way on its own too (a word with its own entry keeps that entry). If the IPA isn't one word
  per written word, the phrase falls back to an `<alias>` with its respelling.
- Graphemes are case-sensitive in all three engines: an all-lowercase lexicon key matches any case in `vc.py`
  but only that spelling in the dictionary.
- Azure: run its lexicon validation tool before production; it wants decomposed Unicode, which `vc.py` does not
  enforce (not verified whether composed IPA is refused).

## Other engines (no format of their own yet)

Read in each vendor's docs on 2026-11-11 through a summarising fetcher, so check the literal syntax on the linked
page before relying on it. "Use" says what to give the engine from this skill today.

| Engine | Takes | Use | Source |
|---|---|---|---|
| Deepgram Aura-2 / Flux TTS | inline JSON with IPA: `{"word": "Azathioprine", "pronounce": "æzəˈθaɪəpɹiːn"}` (≤ 500 per request; early access on Flux) | `text`, or hand-wrap the lexicon IPA | developers.deepgram.com/docs/tts-voice-controls |
| Inworld TTS-2 / TTS-2 Flash | inline IPA in slashes, one word at a time: `Welcome to /kriːt/`; dictionaries (TTS-2) | `elevenlabs` (same bare-slash syntax; one word per entry) | docs.inworld.ai/docs/tts/capabilities/custom-pronunciation |
| MiniMax speech-2.8 | `pronunciation_dict.tone`: `"resume/(rɪˈzjuːm)"` or `"omg/oh my god"`; inline IPA in parentheses `(lɪv)` | `text`, or put lexicon entries in `tone` | platform.minimax.io/docs/api-reference/speech-t2a-http |
| Fish Audio s2.1-pro | CMU Arpabet: `<\|phoneme_start\|>EH1 N JH AH0 N IH1 R<\|phoneme_end\|>`; dictionaries (`phoneme` or `alias` rules) | `text` (IPA is accepted by its drama-3 models only) | docs.fish.audio/developer-guide/core-features/fine-grained-control/english |
| Rime mistv3 / coda | mistv3: Rime's own alphabet in braces `{k1Ast0xm}` with `phonemizeBetweenBrackets: true`; coda: `customLexicon` in X-SAMPA | `text` | docs.rime.ai/platform/pronunciation-control |
| Murf Falcon 2.1 / Gen2 | inline `"/mɜːrf/"` (Falcon 2.1); `pronunciationDictionary` with `"type": "IPA"` | `elevenlabs`, adding the double quotes; or the dictionary | murf.ai/api/docs/text-to-speech/speech-customization |
| xAI TTS | `replace` map: a respelling, or IPA in slashes `"nginx": "/ˈɛndʒɪn ˈɛks/"` | `text`, or put lexicon entries in `replace` | docs.x.ai/developers/model-capabilities/audio/text-to-speech |
| Speechify simba-3.5 | SSML `<sub alias>`; `pronunciations` field (`ipa` or `alias`), IPA "except English" | `text` | docs.speechify.ai/build/guides/text-to-speech/pronunciation |
| Smallest.ai Lightning v3.1 | dictionaries of respellings ("not IPA") | `text` | docs.smallest.ai/models/documentation/text-to-speech-lightning/pronunciation-dictionaries |
| Google Gemini TTS | style prompt and bracket tags; the docs mention no SSML, phonemes or custom pronunciations | `text` | docs.cloud.google.com/text-to-speech/docs/gemini-tts |
| IndexTTS-2.5 (open) | inline `<minute\|M IH1 . N AH0 T>` (CMU), `<行\|XING2>` (pinyin) | `text` | github.com/index-tts/index-tts |

## Picking a format

1. If the engine accepts IPA (SSML phoneme, inline IPA, PLS phoneme), use it for every lexicon entry with `ipa`:
   it is more exact than any respelling, especially for foreign sounds.
2. Otherwise use `text`. Respellings are where most of the iteration happens: if `check` reports a name heard
   wrong, try the next candidate from `vc.py respell`, and re-check.
3. For engines with dictionaries (ElevenLabs, Azure, Polly), `export-pls` once and attach the dictionary: the
   on-screen script and captions stay untouched, and every future script benefits.
4. Whatever the format, `check` judges the audio, not the input: it is the only proof the engine obeyed.

Not verified: what Polly generative's "partial" phoneme support covers; whether inline slashes work on ElevenLabs
v3/v4 Turbo, and whether v4 needs the double quotes around them; Google customPronunciations locale coverage (the
API reference says "initial support is for en-us", the Chirp 3 HD page lists 27 locales without it); whether
Azure's MAI-Voice-2.1 voices take `<phoneme>` (its page shows only `mstts:express-as`); what Google and Polly do
with a phone outside their en-US tables. Hume's Octave, PlayHT and Resemble's hosted API appear to be
discontinued (vendor pages unreachable or saying so): not listed.
