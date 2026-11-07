#!/usr/bin/env python3
"""Flag typographic Unicode in prose: dashes, arrows, curly quotes, ellipsis.

Markdown table rows are skipped. Add `<!-- skip-lint -->` or `# skip-lint` to a
line to skip it.
"""

import re
import sys

from _prose import collect_paths, prose_lines, skipped

CHECKS = {
    "em or en dash, use a conjunction or a full stop": r"[—–]",
    "Unicode arrow, use -> or -->": r"[→←⇒⇐↔]",
    "curly quote, use a straight one": r"[“”‘’]",
    "ellipsis character, use ...": r"…",
}
# two spaces between words, the residue of a deleted dash (markdown only, since
# code aligns inline comments with runs of spaces)
DOUBLE_SPACE = r"[a-z,)]  [a-z(]"

failures = []
for p in collect_paths(sys.argv[1:]):
    for n, line in prose_lines(p):
        if skipped(line) or line.lstrip().startswith("|"):
            continue
        for msg, pattern in CHECKS.items():
            if re.search(pattern, line):
                failures.append((p, n, msg, line.strip()))
        if p.suffix == ".md" and re.search(DOUBLE_SPACE, line):
            failures.append((p, n, "double space mid-sentence", line.strip()))

if failures:
    print("The following lines contain typographic punctuation:")
    for p, n, msg, line in failures:
        print(f"{p}:{n}: {msg}\n\t{line}")
    sys.exit(1)
