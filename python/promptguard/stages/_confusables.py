"""Vendored subset of Unicode confusables for ASCII lookalike detection.

Derived from the Unicode Consortium's confusables.txt
(https://unicode.org/Public/security/latest/confusables.txt).

Only characters that visually resemble ASCII letters are included — the set
relevant to prompt-injection homoglyph attacks.  Fullwidth/compatibility
variants are already handled by NFKC normalisation before this map is applied.
"""

# Maps non-ASCII codepoint → its ASCII lookalike.
CONFUSABLES: dict[str, str] = {
    # ------------------------------------------------------------------ #
    # Cyrillic → Latin lowercase
    # ------------------------------------------------------------------ #
    "а": "a",  # а  CYRILLIC SMALL LETTER A
    "е": "e",  # е  CYRILLIC SMALL LETTER IE
    "о": "o",  # о  CYRILLIC SMALL LETTER O
    "р": "p",  # р  CYRILLIC SMALL LETTER ER
    "с": "c",  # с  CYRILLIC SMALL LETTER ES
    "у": "y",  # у  CYRILLIC SMALL LETTER U
    "х": "x",  # х  CYRILLIC SMALL LETTER HA
    "і": "i",  # і  CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    "ѕ": "s",  # ѕ  CYRILLIC SMALL LETTER DZE
    "һ": "h",  # ħ  CYRILLIC SMALL LETTER SHHA
    "ӏ": "l",  # ӏ  CYRILLIC SMALL LETTER EL WITH TAIL
    # ------------------------------------------------------------------ #
    # Cyrillic → Latin uppercase
    # ------------------------------------------------------------------ #
    "А": "A",  # А  CYRILLIC CAPITAL LETTER A
    "В": "B",  # В  CYRILLIC CAPITAL LETTER VE
    "С": "C",  # С  CYRILLIC CAPITAL LETTER ES
    "Е": "E",  # Е  CYRILLIC CAPITAL LETTER IE
    "Н": "H",  # Н  CYRILLIC CAPITAL LETTER EN
    "І": "I",  # І  CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
    "Ј": "J",  # Ј  CYRILLIC CAPITAL LETTER JE
    "К": "K",  # К  CYRILLIC CAPITAL LETTER KA
    "М": "M",  # М  CYRILLIC CAPITAL LETTER EM
    "О": "O",  # О  CYRILLIC CAPITAL LETTER O
    "Р": "P",  # Р  CYRILLIC CAPITAL LETTER ER
    "Т": "T",  # Т  CYRILLIC CAPITAL LETTER TE
    "У": "Y",  # У  CYRILLIC CAPITAL LETTER U
    "Х": "X",  # Х  CYRILLIC CAPITAL LETTER HA
    # ------------------------------------------------------------------ #
    # Greek → Latin lowercase
    # ------------------------------------------------------------------ #
    "α": "a",  # α  GREEK SMALL LETTER ALPHA
    "ε": "e",  # ε  GREEK SMALL LETTER EPSILON
    "ι": "i",  # ι  GREEK SMALL LETTER IOTA
    "ν": "v",  # ν  GREEK SMALL LETTER NU
    "ο": "o",  # ο  GREEK SMALL LETTER OMICRON
    "ρ": "p",  # ρ  GREEK SMALL LETTER RHO
    "υ": "u",  # υ  GREEK SMALL LETTER UPSILON
    "χ": "x",  # χ  GREEK SMALL LETTER CHI
    # ------------------------------------------------------------------ #
    # Greek → Latin uppercase
    # ------------------------------------------------------------------ #
    "Α": "A",  # Α  GREEK CAPITAL LETTER ALPHA
    "Β": "B",  # Β  GREEK CAPITAL LETTER BETA
    "Ε": "E",  # Ε  GREEK CAPITAL LETTER EPSILON
    "Ζ": "Z",  # Ζ  GREEK CAPITAL LETTER ZETA
    "Η": "H",  # Η  GREEK CAPITAL LETTER ETA
    "Ι": "I",  # Ι  GREEK CAPITAL LETTER IOTA
    "Κ": "K",  # Κ  GREEK CAPITAL LETTER KAPPA
    "Μ": "M",  # Μ  GREEK CAPITAL LETTER MU
    "Ν": "N",  # Ν  GREEK CAPITAL LETTER NU
    "Ο": "O",  # Ο  GREEK CAPITAL LETTER OMICRON
    "Ρ": "P",  # Ρ  GREEK CAPITAL LETTER RHO
    "Τ": "T",  # Τ  GREEK CAPITAL LETTER TAU
    "Υ": "Y",  # Υ  GREEK CAPITAL LETTER UPSILON
    "Χ": "X",  # Χ  GREEK CAPITAL LETTER CHI
    # ------------------------------------------------------------------ #
    # Other script lookalikes
    # ------------------------------------------------------------------ #
    "ℓ": "l",  # ℓ  SCRIPT SMALL L
    "ɩ": "i",  # ɩ  LATIN SMALL LETTER IOTA
    "ɑ": "a",  # ɑ  LATIN SMALL LETTER ALPHA
    "ɡ": "g",  # ɡ  LATIN SMALL LETTER SCRIPT G
    "ᴀ": "A",  # ᴀ  LATIN LETTER SMALL CAPITAL A
    "ʙ": "B",  # ʙ  LATIN LETTER SMALL CAPITAL B
    "ᴄ": "C",  # ᴄ  LATIN LETTER SMALL CAPITAL C
    "ᴅ": "D",  # ᴅ  LATIN LETTER SMALL CAPITAL D
    "ᴇ": "E",  # ᴇ  LATIN LETTER SMALL CAPITAL E
    "ɪ": "I",  # ɪ  LATIN LETTER SMALL CAPITAL I
    "ᴋ": "K",  # ᴋ  LATIN LETTER SMALL CAPITAL K
}
