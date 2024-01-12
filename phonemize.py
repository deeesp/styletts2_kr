from phonemizer.backend import BACKENDS
from phonemizer.separator import Separator

word_separator = " "
syllable_separator = ""
phone_separator = ""
backend="espeak"

separator = Separator(
    word=word_separator,
    syllable=syllable_separator,
    phone=phone_separator,
)

phonemizer_ko = BACKENDS[backend](
    language="ko",
    with_stress=False,
    preserve_punctuation=True,
    words_mismatch='ignore',
    language_switch='remove-flags'
)

phonemizer_en = BACKENDS[backend](
    language="en-us",
    with_stress=True,
    preserve_punctuation=True,
    words_mismatch='ignore',
    language_switch='remove-flags'
)

def text2ipa_kor(text):
    tokens = phonemizer_ko.phonemize([text])[0]
    tokens = [c for c in tokens]
    return tokens

def text2ipa_eng(text):
    tokens = phonemizer_en.phonemize([text])[0]
    tokens = [c for c in tokens]
    return tokens