from voicecheck import text as T


def readings(s):
    return [p for p in T.parts(s) if isinstance(p, tuple)]


def test_norm_folds_accents_and_strips_quotes():
    assert T.norm("Søren, Łukasz and Szegedy") == ["soren", "lukasz", "and", "szegedy"]
    assert T.norm("the 'gate' isn't open") == ["the", "gate", "isn't", "open"]
    assert T.norm("ReLU-based C++ & R") == ["relu", "based", "c", "plus", "plus", "and", "r"]


def test_year_reads_as_a_year_first_but_keeps_the_count():
    (r,) = readings("in 1959")
    assert r[0] == "nineteen fifty nine"
    assert "one thousand nine hundred fifty nine" in r
    assert T.words("in 1959") == ["in", "nineteen", "fifty", "nine"]


def test_numbers_have_readings_with_and_without_and():
    (r,) = readings("784 weights")
    assert r[0] == "seven hundred eighty four"
    assert "seven hundred and eighty four" in r
    assert "seven eight four" in r


def test_grouped_thousands():
    assert T.words("13,000,000 people") == ["thirteen", "million", "people"]
    assert T.words("1,234") == "one thousand two hundred thirty four".split()


def test_hyphens_between_letters_and_digits_are_not_minus():
    assert T.words("GPT-4 and ResNet-50 and COVID-19") == ["gpt", "four", "and", "resnet", "fifty", "and", "covid",
                                                          "nineteen"]


def test_ranges_and_signs():
    assert T.words("pages 10-20") == ["pages", "ten", "to", "twenty"]
    assert ("to", "", "through") in readings("pages 10-20")
    assert T.words("it was -5 out") == ["it", "was", "minus", "five", "out"]


def test_a_hundred():
    (r,) = readings("about 100 people")
    assert "a hundred" in r and r[0] == "one hundred"


def test_money_time_fractions_ordinals_decades_percent():
    assert T.words("$5 million") == ["five", "million", "dollars"]
    assert "four dollars ninety nine" in readings("$4.99")[0]
    assert T.words("at 3:05") == ["at", "three", "oh", "five"]
    assert T.words("10:00") == ["ten", "o'clock"]
    assert T.words("1/2 of 3/4") == ["one", "half", "of", "three", "quarters"]
    assert T.words("the 21st time") == ["the", "twenty", "first", "time"]
    assert T.words("the 1990s and the '80s") == ["the", "nineteen", "nineties", "and", "the", "eighties"]
    assert T.words("90% of 0.5") == ["ninety", "percent", "of", "zero", "point", "five"]
    assert T.words("3.14") == ["three", "point", "one", "four"]
    assert T.words("call 007") == ["call", "zero", "zero", "seven"]


def test_units_glued_to_numbers():
    assert T.words("a 70B model in 3D") == ["a", "seventy", "b", "model", "in", "three", "d"]
    assert ("b", "billion") in readings("70B")


def test_unicode_digits():
    assert T.words("x² and ½") == ["x", "two", "and", "one", "half"]


def test_choose_picks_the_reading_that_fits():
    ps = T.parts("In 1959 we had 784")
    target = "in one thousand nine hundred and fifty nine we had seven hundred and eighty four".split()
    picks = T.choose(ps, lambda ws: -abs(len(ws) - len(target)) + sum(a == b for a, b in zip(ws, target)))
    assert T.expand(ps, picks)[0] == target


def test_expand_reports_which_part_each_word_came_from():
    ps = T.parts("In 1959 we")
    words, owner = T.expand(ps)
    assert words == ["in", "nineteen", "fifty", "nine", "we"]
    assert owner[0] == -1 and owner[-1] == -1 and len({o for o in owner[1:4]}) == 1 and owner[1] >= 0


def test_abbreviations_read_like_the_voice_says_them():
    assert T.words("Dr. Nguyen and Mrs. Smith, etc.") == ["doctor", "nguyen", "and", "missus", "smith", "et", "cetera"]
    assert T.words("Dr Nguyen vs Prof Lee") == ["doctor", "nguyen", "versus", "professor", "lee"]
    assert ("saint", "street") in readings("Main St.")
    assert T.words("e.g. this, i.e. that") == ["for", "example", "this", "that", "is", "that"]
    # acronyms and ordinary words that only look like one are left alone
    assert T.words("MS and ST, vsync, the dr's office") == ["ms", "and", "st", "vsync", "the", "dr's", "office"]


# ---- readings a voice really uses (bug hunt 2026-11-11). A token's first reading is what it was before, so added
# readings never change what a passing take was held to.

def ways(s):
    """Every way the whole text can be read, as strings."""
    import itertools
    opts = [list(p) if isinstance(p, tuple) else [p] for p in T.parts(s)]
    return {" ".join(w for piece in combo for w in T.norm(piece)) for combo in itertools.product(*opts)}


def test_money_with_cents():
    assert T.words("$.50") == ["fifty", "cents"]                              # was a ValueError
    assert "it costs five dollars today" in ways("It costs $5.00 today.")
    assert T.words("$5.00") == ["five", "point", "zero", "zero", "dollars"]   # the old default stays first
    assert "ninety nine cents" in ways("$0.99") and T.words("$0.99") == ["zero", "ninety", "nine"]
    assert "one pound and fifty pence" in ways("£1.50")
    assert "one thousand two hundred thirty four dollars and fifty cents" in ways("$1,234.50")
    assert "ninety nine cents" not in ways("$4.99")


def test_a_scale_needs_to_be_a_scale():
    assert T.words("a $20 T-shirt") == ["a", "twenty", "dollars", "t", "shirt"]          # not "twenty trillion"
    assert T.words("$5 Million was raised") == ["five", "million", "dollars", "was", "raised"]
    assert T.words("$5M and $3bn and $2 billion") == ["five", "million", "dollars", "and", "three", "billion",
                                                     "dollars", "and", "two", "billion", "dollars"]


def test_mixed_fractions():
    assert T.words("2½ hours") == ["two", "and", "one", "half", "hours"]                  # not "twenty one halves"
    assert "two and a half hours" in ways("2½ hours")
    assert T.words("3 1/2 cups") == ["three", "one", "half", "cups"]
    assert "three and a half cups" in ways("3 1/2 cups")


def test_number_sign_degrees_roman_numerals():
    assert T.words("We are #1") == ["we", "are", "one"] and "we are number one" in ways("We are #1")
    assert T.words("No. 5") == ["no", "five"] and "number five" in ways("No. 5")
    assert T.words("#MeToo") == ["metoo"]
    assert T.words("5°C") == ["five", "c"]
    assert {"five degrees celsius", "five degrees c", "five degrees"} <= ways("5°C")
    assert "minus forty degrees fahrenheit" in ways("-40°F") and "ninety degrees" in ways("90°")
    assert T.words("World War II") == ["world", "war", "ii"]
    assert {"world war two", "world war the second"} <= ways("World War II")
    assert "henry the eighth" in ways("Henry VIII") and "louis the fourteenth" in ways("Louis XIV")
    assert ways("the IV drip") == {"the iv drip"}                             # no capitalised word before it


def test_bare_plurals_point_oh_and_clock_times():
    assert "thousands of people" in ways("1000s of people") and "hundreds of" in ways("100s of")
    assert T.words("1000s") == ["one", "thousands"]
    assert T.words("Web 2.0") == ["web", "two", "point", "zero"] and "web two point oh" in ways("Web 2.0")
    assert "three point oh five" in ways("3.05")
    assert T.words("10:30 a.m.") == ["ten", "thirty", "a", "m"]
    assert T.words("10.30am") == ["ten", "point", "three", "zero", "am"]
    assert {"ten thirty am", "ten thirty a m"} <= ways("10.30am") and "three thirty p m" in ways("3.30 pm")
    assert "the one hundred first time" in ways("the 101st time")


def test_long_digit_strings_are_read_digit_by_digit():
    assert T.words("1" * 400) == ["one"] * 400                                # was an OverflowError in num2words
    assert T.words("1" * 5000)[:2] == ["one", "one"]                          # past int()'s digit limit, too


def test_a_tokens_first_reading_is_never_empty():
    # voicecheck/proof.py makes one unit per tuple and reads its first reading
    import unicodedata
    for s in ["#1 No. 5 5°C -5° 90° 10.30am 10:30 p.m. 3 1/2 Henry VIII the IV $5.00 $.50 1000s 2.0 101st $20 T-shirt"]:
        for m in T._TOKEN.finditer(unicodedata.normalize("NFKC", s)):
            for part in T._readings(m):
                assert (isinstance(part, tuple) and T.norm(part[0])) or not T.norm(part)
