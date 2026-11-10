# Respelling for voices that only read text

`vc.py respell IPA [--name]` does this automatically. This page is for checking its output and adjusting by hand.

## Rules that hold across engines

- One token per word, syllables joined by hyphens: `Seg-eh-dee`, not `Seg eh dee` (spaces can make an engine pause
  or read each piece as a word). The spaced form is offered only as a last resort.
- Capitalise at most the stressed syllable of a multi-syllable word (`Shiv-AWN`). Never output a lone all-caps
  word or a chunk that is a known acronym (US, IT, AI, OK, TV, ID): engines spell those letter by letter
  (Cartesia's and misaki's docs say so), and ElevenLabs v4 reads capitals as emphasis. The default `plain` style
  capitalises only a name's first letter; try `stress` if the voice puts the stress in the wrong place.
- Write schwa as "uh" (`Sur-shuh`): no engine reads "ə".
- Avoid syllables that spell a different real word (Wikipedia's warning: "MOW-ee" for Maui).
- Keep g hard before e/i/y with "gh" (`Ghee-zuh`), and s unvoiced with "ss" where it could read as z.
- A stressed short vowel takes the next consonant (`Seg-eh-dee`, `Jef-ree`), except p/t/k before a stressed vowel,
  which start the next syllable (`puh-TAH-toh`).

## IPA → respelling (Wikipedia's key, Help:Pronunciation_respelling_key, with TTS adjustments)

Vowels: æ a · ɑː ah · ɛər air · ɑːr ar · ær arr · ɔː aw · eɪ ay · ɛ e (eh at a syllable's end) · iː, i ee ·
ɪər eer · ɛr err · juː ew / yoo · aɪ y / eye · ɪ i (ih at a syllable's end) · aɪər ire · ɒ o · oʊ oh · ɔɪ oy ·
uː oo · ʊər oor · ɔːr or · aʊ ow · ʌ u (uh at a syllable's end) · ɜːr ur · ʊ uu · ə uh · ər er

Consonants: tʃ ch · ð dh · ɡ g (gh before e/i/y) · dʒ j · x kh · ŋ ng · ʃ sh · θ th · ʒ zh · j y · hw wh ·
the rest as written.

Foreign sounds (common anglicisations, not from Wikipedia's key): ɲ ny · ʎ ly · ø, œ ur · y ew · ç kh ·
ɣ g · ʁ ʀ ɾ r r · ɬ hl · β v · ɕ ʂ sh · ʑ ʐ zh · ɨ ih · ɯ oo · pure e / o ay / oh · ɐ uh ·
nasal ɑ̃ ɛ̃ ɔ̃ œ̃ ahn / an / on / un · ʔ dropped · length and tones dropped.

## When the respelling still comes out wrong

1. Re-check with `vc.py check`; read `heard` against `target` in the `names` entry, sound by sound.
2. Change only the syllable that went wrong: a wrong vowel usually needs a less ambiguous spelling (ay vs ai, oo vs
   u); a wrong stress needs the `stress` candidate, or splitting the word differently.
3. If the engine can take IPA in any form, switch that word to IPA instead of fighting the respelling.
