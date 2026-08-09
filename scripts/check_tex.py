# Copyright (c) 2026 WHACT. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

"""Quick validation of framework.tex for compilation readiness."""
import re
from collections import Counter
from pathlib import Path

tex_path = Path(__file__).resolve().parent.parent / "docs" / "framework.tex"
text = tex_path.read_text(encoding="utf-8")
lines = text.split("\n")

print(f"Total lines: {len(lines)}")
print()

# 1. Non-ASCII
print("=== Non-ASCII Characters ===")
non_ascii = []
for i, line in enumerate(lines, 1):
    for j, ch in enumerate(line):
        if ord(ch) > 127:
            non_ascii.append((i, j, ch, hex(ord(ch))))
if non_ascii:
    for ln, col, ch, hx in non_ascii:
        print(f"  Line {ln}, col {col}: {repr(ch)} ({hx})")
else:
    print("  None found.")
print()

# 2. Brace balance
print("=== Brace Balance ===")
depth = 0
issues = []
for i, line in enumerate(lines, 1):
    for j, ch in enumerate(line):
        if ch == "{" and (j == 0 or line[j - 1] != "\\"):
            depth += 1
        elif ch == "}" and (j == 0 or line[j - 1] != "\\"):
            depth -= 1
            if depth < 0:
                issues.append(f"  Underflow at line {i}, col {j}")
                depth = 0
if issues:
    for iss in issues:
        print(iss)
print(f"  Final balance: {depth} (should be 0)")
print()

# 3. Environment matching
print("=== Environment Matching ===")
begins = re.findall(r"\\begin\{(\w+)\}", text)
ends = re.findall(r"\\end\{(\w+)\}", text)
bc = Counter(begins)
ec = Counter(ends)
all_envs = sorted(set(list(bc.keys()) + list(ec.keys())))
for env in all_envs:
    b, e = bc.get(env, 0), ec.get(env, 0)
    status = "OK" if b == e else "MISMATCH"
    print(f"  {status}: {env} (begin={b}, end={e})")
print()

# 4. Markdown remnants
print("=== Markdown Remnants ===")
md_issues = []
for i, line in enumerate(lines, 1):
    s = line.strip()
    if "$$" in s:
        md_issues.append((i, "Contains $$ (markdown math)"))
    if s.startswith("## ") or s.startswith("### "):
        md_issues.append((i, "Markdown heading"))
    if re.search(r"^\s*[-*]\s+\S", s) and "\\item" not in s:
        md_issues.append((i, "Possible markdown bullet"))
if md_issues:
    for ln, desc in md_issues:
        print(f"  Line {ln}: {desc}")
else:
    print("  None found.")
print()

# 5. Potential wide equations
print("=== Potentially Wide Equations ===")
in_eq = False
eq_start = 0
eq_content = []
for i, line in enumerate(lines, 1):
    s = line.strip()
    if s == "\\begin{equation}":
        in_eq = True
        eq_start = i
        eq_content = []
    elif s == "\\end{equation}" and in_eq:
        in_eq = False
        full = " ".join(eq_content)
        if len(full) > 120:
            print(f"  Line {eq_start}: {len(full)} chars - {full[:80]}...")
    elif in_eq:
        eq_content.append(s)
print()

# 6. Section structure
print("=== Section Structure ===")
for i, line in enumerate(lines, 1):
    s = line.strip()
    if (
        s.startswith("\\section{")
        or s.startswith("\\subsection{")
        or s.startswith("\\subsubsection")
    ):
        print(f"  Line {i}: {s[:80]}")
print()

# 7. Cross-references check
print("=== Hardcoded Section References ===")
refs = re.findall(r"\\S\s*[\d.]+", text)
if refs:
    for r in sorted(set(refs)):
        print(f"  {r}")
else:
    print("  None found.")
