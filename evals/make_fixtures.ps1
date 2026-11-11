# Writes the audio the eval cases need (Windows only; uses the built-in Zira voice). Run from the skill folder:
#   powershell -File evals/make_fixtures.ps1
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice("Microsoft Zira Desktop")
$s.SetOutputToWaveFile((Join-Path $PSScriptRoot "judges-a-misread-name\files\take.wav"))
$s.Speak("Ask Siobhan about it.")
$s.Dispose()
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice("Microsoft Zira Desktop")
$s.SetOutputToWaveFile((Join-Path $PSScriptRoot "proofreads-a-transcript\files\meeting.wav"))
$s.Speak("Thanks everyone for joining. Shiv-awn will present the results from 1959. The model is confident in wrong answers, and it goes from zero to two hundred quickly. Christian Seg-eh-dee wrote the paper, and Shiv-awn checked it twice.")
$s.Dispose()
