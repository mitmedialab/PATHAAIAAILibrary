"""Small text helpers shared by the simulator and the retrieval functions."""

import re

# Function words: the little words that carry grammar rather than meaning.
STOPWORDS = set(
    """
    a an the and or but if then else so of to in on at by for with from into onto
    about as is are was were be been being am do does did done have has had having
    i me my mine you your yours he him his she her hers it its we us our ours they
    them their theirs this that these those there here what which who whom whose
    when where why how not no nor can could will would shall should may might must
    just also very too than such own same other some any each every all both few
    more most much many up down out over under again once only s t don doesn didn
    isn aren wasn weren won wouldn shouldn couldn let lets please tell give get
    """.split()
)

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "dozen": 12, "a": 1, "an": 1,
}

_WORD_RE = re.compile(r"[a-z0-9']+")
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def words(text):
    """Lower-case words, apostrophes kept: "Don't panic!" -> ["don't", "panic"]."""
    return _WORD_RE.findall(str(text).lower())


def stem(word):
    """A deliberately crude stemmer: "refunds" -> "refund", "boxes" -> "box"."""
    for suffix in ("ies", "es", "s"):
        if len(word) > 4 and word.endswith(suffix) and not word.endswith("ss"):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def content_words(text):
    """Stemmed words with the function words dropped."""
    return [stem(w.replace("'", "")) for w in words(text) if w not in STOPWORDS]


def numbers(text):
    """Every number in the text, digits or words: "three apples and 12 pears" -> [3, 12]."""
    found = []
    for token in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?|[A-Za-z]+", str(text)):
        low = token.lower()
        if _NUMBER_RE.fullmatch(token):
            value = float(token.replace(",", ""))
            found.append(int(value) if value.is_integer() else value)
        elif low in WORD_NUMBERS and low not in ("a", "an"):
            found.append(WORD_NUMBERS[low])
    return found


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]
