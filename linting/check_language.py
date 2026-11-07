#!/usr/bin/env python3
"""Flag words and phrases the psyphy prose rules ask us to avoid.

ERRORS fail the hook. WORDS are judgment calls and TERMS are technical names that
need a plain gloss in front of them. Both print a reminder but never fail, so run
that hook with `verbose: true` to see them. Add `<!-- skip-lint -->` or
`# skip-lint` to a line to skip it.
"""

import re
import sys

from _prose import collect_paths, prose_lines, skipped

ERRORS = {
    r"\bthe user\b": "say who: a caller, a scientist, or address the reader",
    r"\b([Dd]eliberate(ly)?|on purpose|[Ii]ntentional(ly)?)\b": ("really necessary? "),
    r"\bMode [A-Z]\b|\b(Tier|Phase) [0-9A-Z]\b": "invented label, name the thing",
}

# word -> question to ask. Some hits are right to keep.
WORDS = {
    r"\b(seamless(ly)?|lightweight|canonical|elegant(ly)?|makes it easy)\b": (
        "sales adjective, say what it does instead?"
    ),
    r"\b(crucially|worth internalising|the whole point)\b": (
        "announced significance, really necessary? cut it?"
    ),
}

# term -> plain phrase to lead with. The term may appear once per file inside
# brackets, after the plain phrase.
TERMS = {
    r"Mahalanobis-1": "the one-standard-deviation ellipse (Mahalanobis-1 contour)",
    r"\b[Dd]elta (distribution|function)\b|\b[Dd]irac\b": (
        "point estimate, with no bracket"
    ),
}


def in_brackets(line, start):
    before = line[:start]
    return before.count("(") > before.count(")")


errors = []
reminders = []
questions = []
for p in collect_paths(sys.argv[1:]):
    bracketed = set()
    for n, line in prose_lines(p):
        if skipped(line):
            continue
        for pattern, msg in ERRORS.items():
            if m := re.search(pattern, line):
                errors.append((p, n, m.group(0), msg))
        for pattern, msg in WORDS.items():
            if m := re.search(pattern, line):
                questions.append((p, n, m.group(0), msg))
        for pattern, plain in TERMS.items():
            for m in re.finditer(pattern, line):
                if in_brackets(line, m.start()) and pattern not in bracketed:
                    bracketed.add(pattern)
                    continue
                reminders.append((p, n, m.group(0), plain))

if questions:
    print("Words to reconsider (not an error):")
    for p, n, word, msg in questions:
        print(f"{p}:{n}: {word!r}, {msg}")
    print()
if reminders:
    print("Technical terms without a plain phrase in front (not an error):")
    for p, n, term, plain in reminders:
        print(f"{p}:{n}: {term!r}, lead with {plain!r}")
    print()
if errors:
    print("The following lines use words the prose rules avoid:")
    for p, n, word, msg in errors:
        print(f"{p}:{n}: {word!r}, {msg}")
    sys.exit(1)
