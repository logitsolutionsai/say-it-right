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
