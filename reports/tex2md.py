#!/usr/bin/env python3
"""Render the docs/*.tex report family as Markdown.

Not a general LaTeX converter -- it handles exactly the subset this repo's
reports use (article + booktabs tabular + itemize/enumerate + \\code and
friends), and it is loud about anything it does not understand rather than
silently dropping it.

    python3 reports/tex2md.py docs/FOO.tex [...]     # write docs/FOO.md
    python3 reports/tex2md.py --check docs/*.tex     # report-only, no writes

`--check` lists every macro that survived to the output, so the honest
workflow is: check, extend the tables below, check again, then write.

Display math, tikz and the cgcirc circuit environment cannot be rendered as
text. They become a fenced block of the original TeX or an explicit pointer to
the PDF -- never a silent drop.
"""
import re
import sys
from pathlib import Path

OPAQUE = {"tikzpicture", "cgcirc"}

# Formatting-only: drop the macro, keep nothing.
DROP = r"""small footnotesize scriptsize large Large LARGE normalsize centering
    noindent nopagebreak hfill hfil itshape bfseries rmfamily ttfamily
    bigskip medskip smallskip toprule midrule bottomrule hline linewidth
    textwidth columnwidth raggedright raggedleft protect displaystyle
    nolimits limits left right big Big bigl bigr Bigl Bigr quad qquad""".split()

# Symbol macros, longest name first so \leq does not eat \le.
SYM = {
    r"\textasciitilde": "~", r"\rightarrow": "→", r"\leftarrow": "←",
    r"\bigoplus": "⊕", r"\mathbb": "", r"\mathrm": "", r"\mathit": "",
    r"\mathbf": "", r"\mathrel": "", r"\mathbin": "", r"\operatorname": "",
    r"\approx": "≈", r"\lnot": "¬", r"\langle": "⟨", r"\rangle": "⟩",
    r"\mapsto": "↦", r"\oplus": "⊕", r"\times": "×", r"\ldots": "…",
    r"\dots": "…", r"\cdot": "·", r"\prod": "∏", r"\sum": "∑",
    r"\geq": "≥", r"\leq": "≤", r"\neq": "≠", r"\cup": "∪", r"\cap": "∩",
    r"\alpha": "α", r"\beta": "β", r"\sigma": "σ", r"\rho": "ρ",
    r"\Delta": "Δ", r"\delta": "δ", r"\mu": "μ", r"\ell": "ℓ",
    r"\flat": "♭", r"\hat": "^", r"\pm": "±", r"\in": "∈", r"\mid": "|",
    r"\to": "→", r"\ge": "≥", r"\le": "≤", r"\S": "§", r"\xor": "⊕",
    r"\vee": "∨", r"\wedge": "∧", r"\lor": "∨", r"\land": "∧",
    r"\subseteq": "⊆", r"\emptyset": "∅", r"\infty": "∞",
}

# One-argument macros -> a Markdown wrapper.
WRAP = {
    "code": "`{}`", "texttt": "`{}`", "verb": "`{}`", "path": "`{}`",
    "textbf": "**{}**", "emph": "*{}*", "textit": "*{}*", "textsf": "{}",
    "text": "{}", "mbox": "{}", "textrm": "{}", "textsc": "{}",
    "fbox": "{}", "framebox": "{}", "caption": "{}", "footnote": " ({})",
    "paragraph": "**{}**", "overline": "{}", "underline": "{}",
}
# Two-argument macros. The format string sees both, in order.
TWO = {"frac": "{}/{}", "tfrac": "{}/{}", "dfrac": "{}/{}",
       "parbox": "{1}", "minipage": "{1}", "textcolor": "{1}"}
# One-argument macros whose argument is discarded.
EAT = ["label", "ref", "eqref", "cite", "index", "vspace", "hspace", "phantom"]

# Escapes that must be neutralised BEFORE the macro scanner runs. Otherwise
# `\{` is read as the macro-less backslash plus a brace, the two are separated,
# and the later \{ -> { replacement no longer matches -- which is exactly how
# set-builder notation such as \{v_x, v_t\} used to leave stray \v and \h in
# the output.
ESCAPES = {r"\{": "\x01", r"\}": "\x02", r"\%": "\x03", r"\_": "\x04",
           r"\&": "\x05", r"\#": "\x06", r"\$": "\x07",
           # {,} is the thousands separator and -{}- is a literal double dash
           # (the {} stops the en-dash ligature). Both must survive the dash
           # rewriting below, so they ride as placeholders too.
           "{,}": ",", "-{}-": "\x0b"}
UNESCAPE = {"\x01": "{", "\x02": "}", "\x03": "%", "\x04": "_",
            "\x05": "&", "\x06": "#", "\x07": "$", "\x0b": "--"}
THINSPACE = [r"\,", r"\;", r"\!", r"\:", r"\ "]


def brace_arg(s: str, i: int) -> tuple[str, int]:
    """Read a balanced {...} starting at s[i]=='{'. Returns (inner, end)."""
    depth, start = 1, i + 1
    j = start
    while j < len(s) and depth:
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
        j += 1
    return s[start:j - 1], j


def apply_macros(s: str) -> str:
    """Rewrite \\name{...} for every macro we know, innermost-last.

    Regex cannot do this: \\code{\\textbf{x}} has nested braces, and the
    naive [^{}]* pattern silently leaves the outer macro behind. We scan and
    brace-match instead, recursing into each argument.
    """
    out, i = [], 0
    while i < len(s):
        if s[i] != "\\":
            out.append(s[i]); i += 1; continue
        m = re.match(r"\\([a-zA-Z]+)\*?", s[i:])
        if not m:
            out.append(s[i]); i += 1; continue
        name, after = m.group(1), i + m.end()
        # Skip an optional [..] argument.
        if after < len(s) and s[after] == "[":
            close = s.find("]", after)
            if close > 0:
                after = close + 1
        has_arg = after < len(s) and s[after] == "{"
        if name in EAT and has_arg:
            _, i = brace_arg(s, after); continue
        if name in TWO and not has_arg:
            # Brace-less arguments, as in \tfrac23. Take two single tokens.
            toks = s[after:after + 2]
            if len(toks) == 2 and toks.isalnum():
                out.append(f"{toks[0]}/{toks[1]}"); i = after + 2; continue
            i = after; continue
        if name in TWO and has_arg:
            a1, j = brace_arg(s, after)
            if j < len(s) and s[j] == "{":
                a2, i = brace_arg(s, j)
                out.append(TWO[name].format(apply_macros(a1), apply_macros(a2)))
                continue
            # Only one argument present -- fall through to a plain unwrap.
            out.append(apply_macros(a1)); i = j; continue
        if name in WRAP and has_arg:
            inner, i = brace_arg(s, after)
            out.append(WRAP[name].format(apply_macros(inner))); continue
        if name in DROP:
            i = after
            if has_arg:
                inner, i = brace_arg(s, after)
                out.append(apply_macros(inner))
            continue
        out.append(s[i:after]); i = after
    return "".join(out)


def inline(s: str) -> str:
    s = re.sub(r"\\href\{([^{}]*)\}\{([^{}]*)\}", r"[\2](\1)", s)
    s = re.sub(r"\\url\{([^{}]*)\}", r"<\1>", s)
    for k, v in ESCAPES.items():
        s = s.replace(k, v)
    for t in THINSPACE:
        s = s.replace(t, " ")
    s = apply_macros(s)
    for k in sorted(SYM, key=len, reverse=True):
        s = s.replace(k, SYM[k])

    def demath(m):
        inner = apply_macros(m.group(1))
        for k in sorted(SYM, key=len, reverse=True):
            inner = inner.replace(k, SYM[k])
        inner = re.sub(r"[_^]\{([^{}]*)\}", r"_\1", inner)
        inner = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", inner)
        return re.sub(r"\\[a-zA-Z]+", "", inner).replace("{", "").replace("}", "").strip()

    s = re.sub(r"\$([^$]*)\$", demath, s)
    # Dashes BEFORE unescaping, or the -{}- placeholder's restored "--" gets
    # eaten by the en-dash rule and `--gss` renders as `–gss`.
    s = s.replace("---", "—").replace("--", "–")
    for k, v in UNESCAPE.items():
        s = s.replace(k, v)
    s = s.replace("``", '"').replace("''", '"').replace("~", " ")
    # \\ and \\[4pt] are line breaks; Markdown gets the break for free.
    s = re.sub(r"\\\\\[[^\]]*\]", " ", s)
    s = re.sub(r"\\\\\s*$", "", s)
    return re.sub(r"[ \t]+", " ", s).strip()


def table(body: str) -> list[str]:
    rows, ncol = [], 0
    for raw in body.split("\\\\"):
        line = re.sub(r"\\cmidrule\(?[lr]*\)?\{[^}]*\}", "", raw.strip())
        line = re.sub(r"\\(?:top|mid|bottom)rule", "", line)
        if not line.strip():
            continue
        cells = []
        for c in line.split("&"):
            c = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{(.*)\}", r"\1", c.strip(), flags=re.S)
            cells.append(inline(c))
        if not any(cells):
            continue
        ncol = max(ncol, len(cells))
        rows.append(cells)
    if not rows:
        return []
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    return (["| " + " | ".join(rows[0]) + " |", "|" + "---|" * ncol]
            + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def figure(m):
    inner = m.group(0)
    img = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]*)\}", inner)
    cap = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|\\end|$)", inner, re.S)
    alt = inline(cap.group(1)) if cap else "figure"
    if not img:
        return "\n[figure — see the PDF]\n"
    src = img.group(1)
    if not src.lower().endswith((".png", ".jpg", ".pdf")):
        src += ".png"
    return f"\n![{alt}]({src})\n\n*{alt}*\n"


def convert(tex: str, stem: str) -> tuple[str, list[str]]:
    warn: list[str] = []
    tex = re.sub(r"(?<!\\)%.*$", "", tex, flags=re.M)
    tex = re.sub(r"\\(?:" + "|".join(DROP) + r")(?![a-zA-Z])", "", tex)

    # Expand the document's own \newcommands before anything else, or they
    # survive as unknown macros (\fmix, \CNOT, \xor, ...). Scan the WHOLE
    # file, not just the preamble: NONLINEAR_RG_CG_MENU defines \cgvar and
    # \note in the body, right above their first use.
    for name, arity, defn in re.findall(r"\\newcommand\{?\\([a-zA-Z]+)\}?(\[\d+\])?\{(.*)\}\s*$",
                                 tex, re.M):
        if arity or "#" in defn:
            WRAP.setdefault(name, "{}")
        else:
            SYM.setdefault("\\" + name, inline(defn))

    title = (re.search(r"\\title\{(.*?)\}\s*$", tex, re.S | re.M) or [None, stem])[1]
    title = inline(re.sub(r"\\\\.*", "", title, flags=re.S))
    date = re.search(r"\\date\{([^{}]*)\}", tex)

    body = tex.split("\\begin{document}", 1)[-1].split("\\end{document}")[0]

    head = [f"# {title}", ""]
    if date and date.group(1).strip():
        head += [f"*{inline(date.group(1))}*", ""]
    head += [f"> Markdown rendering of `docs/{stem}.tex`. The PDF is "
             "authoritative for figures, diagrams and display math.", ""]

    for env in OPAQUE:
        body, n = re.subn(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}",
                          f"\n[{env} diagram — see the PDF]\n", body, flags=re.S)
        if n:
            warn.append(f"{n} {env} environment(s) -> pointer to the PDF")
    body, n = re.subn(r"\\begin\{figure\}.*?\\end\{figure\}", figure, body, flags=re.S)
    if n:
        warn.append(f"{n} figure(s) -> image links")

    def _table(m):
        inner = m.group(0)
        cap = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|\\end)", inner, re.S)
        tb = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", inner, re.S)
        parts = table(tb.group(1)) if tb else []
        if cap:
            parts += ["", f"*{inline(cap.group(1))}*"]
        return "\n" + "\n".join(parts) + "\n"

    body = re.sub(r"\\begin\{table\}(?:\[[^\]]*\])?.*?\\end\{table\}", _table, body, flags=re.S)
    body = re.sub(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}",
                  lambda m: "\n" + "\n".join(table(m.group(1))) + "\n", body, flags=re.S)
    body = re.sub(r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}",
                  lambda m: "\n```\n" + m.group(1).strip("\n") + "\n```\n", body, flags=re.S)
    # Display math stays as TeX in a fence: honest, and readable to anyone who
    # reads TeX. Silently mangling it would be worse.
    body = re.sub(r"\\begin\{(equation|align|gather)\*?\}(.*?)\\end\{\1\*?\}",
                  lambda m: "\n```tex\n" + m.group(2).strip() + "\n```\n", body, flags=re.S)
    body = re.sub(r"\\\[(.*?)\\\]", lambda m: "\n```tex\n" + m.group(1).strip() + "\n```\n",
                  body, flags=re.S)
    body = re.sub(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                  lambda m: "\n## Abstract\n" + m.group(1).strip() + "\n", body, flags=re.S)
    for env in ("center", "minipage", "small", "footnotesize", "flushleft", "appendix", "quote"):
        body = re.sub(rf"\\begin\{{{env}\}}(?:\{{[^}}]*\}}|\[[^\]]*\])*", "", body)
        body = body.replace(f"\\end{{{env}}}", "")

    # Macros first, across the whole body: a \fbox{\parbox{...}{...}} callout
    # spans several lines, and matching braces line-by-line cannot see the end.
    for k, v in ESCAPES.items():
        body = body.replace(k, v)
    for t in THINSPACE:
        body = body.replace(t, " ")
    body = apply_macros(body)

    lines, stack = [], []
    for raw in body.split("\n"):
        line = raw.rstrip()
        m = re.match(r"\s*\\((?:sub)*)section\*?\{(.*)\}\s*$", line)
        if m:
            lines += ["", "#" * (len(m.group(1)) // 3 + 2) + " " + inline(m.group(2)), ""]
            continue
        m = re.match(r"\s*\\paragraph\{(.*?)\}(.*)", line)
        if m:
            lines += ["", f"**{inline(m.group(1))}** {inline(m.group(2))}"]
            continue
        if re.match(r"\s*\\begin\{itemize\}", line):
            stack.append("-"); lines.append(""); continue
        if re.match(r"\s*\\begin\{enumerate\}", line):
            stack.append("1."); lines.append(""); continue
        if re.match(r"\s*\\end\{(itemize|enumerate)\}", line):
            stack and stack.pop(); lines.append(""); continue
        m = re.match(r"\s*\\item\s*(.*)", line)
        if m:
            lines.append(f"{'  ' * max(len(stack) - 1, 0)}{stack[-1] if stack else '-'} {inline(m.group(1))}")
            continue
        if re.match(r"\s*\\(maketitle|tableofcontents|newpage|clearpage|appendix|newcommand|renewcommand)\b", line):
            continue
        if line.startswith(("|", "```", "![", "[")):
            lines.append(line); continue
        lines.append(inline(line))

    md = "\n".join(head) + "\n" + re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"
    # Anything still holding a backslash is something we did not handle.
    survivors = sorted(set(re.findall(r"\\[a-zA-Z]+", re.sub(r"```tex.*?```", "", md, flags=re.S))))
    if survivors:
        warn.append("unhandled: " + " ".join(survivors[:14]))
    return md, warn


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    bad = 0
    for path in args:
        p = Path(path)
        md, warn = convert(p.read_text(), p.stem)
        if not check:
            p.with_suffix(".md").write_text(md)
        print(f"{'CHECK' if check else 'wrote'} {p.with_suffix('.md')}  ({len(md.splitlines())} lines)")
        for w in warn:
            print(f"    ! {w}")
            if w.startswith("unhandled"):
                bad += 1
    return 1 if (check and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
