# Researching how a name is said

## Contents
- [Which pronunciation is "right"](#which-pronunciation-is-right)
- [Sources, best first](#sources-best-first)
- [What `vc.py lookup` asks, and where](#what-vcpy-lookup-asks-and-where)
- [Reading what you find](#reading-what-you-find)
- [Worked examples](#worked-examples)
- [When to stop and ask](#when-to-stop-and-ask)

## Which pronunciation is "right"

Broadcasters settled this long ago, and their rule is the one to follow:

- **The person's own pronunciation wins.** "For personal names and titles, the BBC uses the pronunciation preferred
  by the person(s) concerned. To check such usage, the Unit consults them or near relations or colleagues."
  (Oxford Companion to the English Language, *BBC Pronunciation Unit*). A former head of the Unit puts it as
  "establish who 'owns' a name (the bearer if a person...), and follow them" (linguism.co.uk/?p=39). Voice of
  America: "pronounce a person's name as that person pronounces it" (pronounce.voanews.com/about-methodology.php).
  The same goes for a company's brand and a place's own name.
- **When you can't learn their preference**, anglicise from the native pronunciation, as far as is customary: the
  BBC considers "the phonetics and phonology of the relevant language, the opinions of native speakers on how they
  might expect it to be anglicised, and the ease of pronunciation for our broadcasters" (BBC blog, 2008-04, *How
  to say: Morgan Tsvangirai*); Mandarin names are anglicised systematically and tones dropped. VOA: "don't try to
  sound as if you are fluent in a language if you are not." Use an established English form where one exists
  (Joaquin Phoenix is wah-KEEN; LeCun is lə-KUN in English, not French [ləkœ̃]).
- **Last resort: the language's spelling rules**, written in the lexicon as an inference ("Hungarian sz = s").

Record the choice and its evidence in the lexicon (`source:`, and `note:` for the reasoning), so the user or a later
run can see why "Seg-eh-dee" and not "Zeh-ged-ee".

## Sources, best first

| Source | How to find it | Notes |
|---|---|---|
| The person saying their own name | web search: `"<full name>" pronounce`, `"<full name>" interview`, `how do you say <surname>`; their talks, podcast intros, YouTube, a pronunciation line on their site or profile | The gold standard; put the URL and timestamp in `source`. A video page can't go to `hear`, but if the user supplies the clip, `vc.py hear clip.wav --start S --end E` transcribes it. |
| Native-speaker recordings | `vc.py lookup` (Wikidata P443 on the person and on their name items; Commons titles) | 73% of the 11,925 people with recordings on Wikidata are Lingua Libre volunteers, not the person: a recording shows how a speaker of the language says the *name*, not necessarily how this person does. Check the recording's language. `vc.py hear <url>` gives its IPA. |
| Wikipedia lead IPA and respelling | `vc.py lookup` | English `IPAc-en` and `respell` are the established English form; `{{IPA|xx}}` is the native one. Only about half of well-known people have any; the pronunciation often sits in a footnote (`{{efn}}`), which lookup reads. |
| VOA Pronounce | `vc.py lookup` (pronounce.voanews.com) | ~7,000 newsmakers with a respelling and audio, from VOA's own research. |
| Wiktionary | `vc.py lookup` (each word of the name) | Given names and surnames as words, by language, with UK/US variants; many languages generate IPA from a module, which lookup expands. |
| Paid or account-only services | Forvo (API key, from $2/month non-commercial; terms forbid caching the audio), NameShouts (token; 50 free searches/month), NameCoach (institutional) | Not called by the skill. Worth a manual look for a hard name. CMUdict (keyless) only holds anglicised guesses. |
| Spelling rules | your knowledge of the language | State them as inference, and mark the entry unconfirmed. |

## What `vc.py lookup` asks, and where

Keyless public endpoints, called gently (spaced out, `Retry-After` honoured on 429/503: Wikimedia rate-limits
bursts). The User-Agent carries the project's page and email, as Wikimedia's User-Agent policy asks; set
`VOICECHECK_CONTACT` to a URL or email of yours to use your own.

- **Wikipedia**: the exact title (`action=query&titles=<Name>&redirects=1`), then a search whose hits count as the
  person only if their title has every word of the name. Hits that share only the surname come back as
  `match: "surname only"`; they show how the surname is said (e.g. the physicist Abram Ioffe's
  `{{lang-rus|...|p=ɪˈofɛ}}`), not how this person says it. The lead-section wikitext is parsed for `{{IPAc-en}}`,
  `{{IPA|xx|...}}`, the retired `{{IPA-xx}}`, `{{lang-xx|...|p=IPA}}`, `{{respell}}` and audio, and each
  `IPAc-en` is rendered by Wikipedia (`action=parse&text=...`), because its pieces are aliases (`əːr` is `ɜːr`).
- **Wikidata**: the article's item, or for people without an article a search restricted to humans
  (`srsearch=<Name> haswbstatement:P31=Q5`). P898 (IPA) and P443 (audio, language from qualifier P407) on the
  person and on their given/family-name items (P735/P734), plus name items found by searching each word.
- **Wiktionary**: each word's page; `{{IPA|..}}` lines and generated ones (`{{hu-IPA}}`, `{{es-pr}}`, `{{fr-IPA}}`)
  expanded through the API.
- **Commons**: files whose title is the name ("Xx-Name.ogg", "LL-Q<lang> (<code>)-<speaker>-<name>.wav"), searched
  with `intitle:<word> filetype:audio`; through api.wikimedia.org when commons.wikimedia.org is unreachable (some
  networks reset it; api.wikimedia.org itself is being retired, so this fallback may stop working).
- **Audio links** are direct `upload.wikimedia.org` URLs (md5 path), with licence and author read through
  en.wikipedia.org's shared file repository, so nothing needs commons.wikimedia.org.

`lookup` never chooses: it returns every candidate with where it came from, plus `errors` for sources it couldn't
reach. Expect gaps: of six well-known AI and film names tested (2026-11-10), only three had Wikipedia IPA, and
Szegedy, Ioffe and Karpathy had nothing direct anywhere.

## Reading what you find

- `/slashes/` are phonemic (broad), `[brackets]` phonetic (narrow); either works as a lexicon `ipa`. Keep stress
  marks (ˈ): the name check uses them. Drop tones and length if they get in the way.
- `IPAc-en` is *diaphonemic* English (Help:IPA/English): its post-vowel /r/ is only said in rhotic accents and /hw/
  is plain /w/ for most speakers. The name check already treats both as accent details.
- `{{respell|SUR|shə}}` is Wikipedia's respelling: hyphens between syllables, capitals on the stressed one. For a
  TTS, write ə as "uh" (`Sur-shuh`); `vc.py respell` does this from IPA.
- A recording turned into IPA by `hear` is one speaker, transcribed by a model that marks no stress: compare it
  with the text sources rather than trusting it alone.

## Worked examples

**Saoirse Ronan** (2026-11-10). Wikipedia (in a footnote): `IPAc-en` /ˈsɜːrʃə ˈuːnə ˈroʊnən/, `{{respell|SUR|shə|...}}`,
Irish `{{IPA|ga|ˈsˠiːɾˠʃə}}`; Wikidata P898 `sɜrʃə ˈunə ˈroʊnən` (English); Wiktionary /ˈsɪə(ɹ).ʃə/ and
/ˈsɜː(ɹ).ʃə/; Commons has an Irish Lingua Libre recording of the given name. The English form is the one for
English narration; an interview where she says it would confirm it. Lexicon:
`Saoirse: {say: Sur-shuh, ipa: ˈsɜːrʃə, source: "https://en.wikipedia.org/wiki/Saoirse_Ronan"}`.

**Joaquin Phoenix.** Wikipedia `IPAc-en` /hwɑːˈkiːn/ with a CC0 recording (`En-us-Joaquin.oga`) and respelling
whah-KEEN; Wiktionary lists /wɑˈkiːn/ too. /hw/ is /w/ for most speakers: `Joaquin: {say: Wah-keen, ipa: wɑːˈkiːn}`.

**Yann LeCun.** Wikipedia: English `ləˈkʌn` (respelled lə-KUN) and French `ləkœ̃`. English narration takes the
English form: `LeCun: {say: Luh-kun, ipa: ləˈkʌn}`. (His item also lists the given name André; lookup leaves out
name items that aren't in the query.)

**Sergey Ioffe.** No article on any Wikipedia; his Wikidata item (found by the human-only search; its label uses
Cyrillic look-alike letters) has no pronunciation. The Russian surname is Иоффе, IPA ɪˈofɛ (from the physicist's
article, a related page); OpenMontage's lexicon used the anglicised Yoff-ee, chosen by ear. Search for a talk of
his before settling, and record which you used.

**Christian Szegedy.** No English article, no recordings. Hungarian spelling gives ˈsɛɡɛdi (sz = s, e = ɛ, the
archaic final -y = i, first-syllable stress; compare the city Szeged, [ˈsɛɡɛd], which has a Commons recording).
Lexicon: `Szegedy: {say: Seg-eh-dee, ipa: ˈsɛɡɛdi, source: "Hungarian spelling; unconfirmed against his own"}`.

## When to stop and ask

Ask the user before guessing when the name is central (a guest, the video's subject, a sponsor), the sources
disagree and none is the person's own, or the person isn't public enough to have any source. Offer what you found
("Wikipedia gives SUR-shuh; the Irish is SEER-shuh; which does she use?") rather than an open question.
