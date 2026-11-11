"""say-it-right: make AI voice-overs say every word right, names included. Research how a word is said
(lookup, scan), keep it in a lexicon, render it for the TTS engine, and check each take with three ears:
Whisper, a letter model with no language model, and an IPA phone model (verify, phonemes). The same ears, run
the other way, proofread a transcript against its audio (proof)."""

__version__ = "0.3.0"
