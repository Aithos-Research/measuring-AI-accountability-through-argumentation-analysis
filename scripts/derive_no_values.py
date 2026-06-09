"""Derive main_no_values.tex from main_AI_drafted.tex by deleting only the
value-vocabulary content. Everything else is kept verbatim so the two files
agree on numbers and structure outside the values material.
"""
from pathlib import Path

SRC = Path("C:/Users/arnol/Downloads/PCAIDE_PAPERS_AITHOS (6)/WaltonSchemes/main_AI_drafted.tex")
DST = Path("C:/Users/arnol/Downloads/PCAIDE_PAPERS_AITHOS (6)/WaltonSchemes/main_no_values.tex")


def main():
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # 1) Trim the §1 forward-pointer.
    for i, ln in enumerate(lines):
        if "Phase-2 scheme classifications" in ln and "value vocabulary" in ln:
            lines[i] = ln.replace(
                " (with the value vocabulary that surfaces inside the value-centric\nschemes)",
                "",
            )
        elif "Phase-2 scheme classifications" in ln and "vs.\\ external paired comparison" in ln:
            # multi-line span; just drop the trailing values clause when we
            # encounter it independently.
            pass

    text = "".join(lines)

    # Replace explicit value-vocabulary clauses without touching surrounding
    # text.
    text = text.replace(
        " (with the value vocabulary that surfaces inside the value-centric\nschemes)",
        "",
    )
    text = text.replace(
        ", and the value vocabulary that surfaces inside the value-centric schemes",
        "",
    )
    text = text.replace(
        ", then the value-vocabulary picture, then limitations",
        ", then limitations",
    )
    text = text.replace(
        ", and the full Phase-2 value-extraction outputs",
        "",
    )
    text = text.replace(
        " and three pieces of post-processing applied to the\nPhase-2 value lexicon (Table~\\ref{tab:top-values})",
        "",
    )
    text = text.replace(
        "\n\\S\\ref{sec:res:scores} and three pieces of post-processing applied to the\nPhase-2 value lexicon (Table~\\ref{tab:top-values}).",
        "\n\\S\\ref{sec:res:scores}.",
    )

    # Block-level removals: the paragraph + Table 6, the value-vocabulary
    # discussion subsection, and the value appendix subsections.

    def cut(text, start_marker, end_marker, include_end=True):
        i = text.find(start_marker)
        if i == -1:
            return text
        j = text.find(end_marker, i)
        if j == -1:
            raise RuntimeError(f"end marker not found: {end_marker!r}")
        if include_end:
            j += len(end_marker)
        # also gobble trailing whitespace / single newline
        while j < len(text) and text[j] in ("\n", " "):
            j += 1
        return text[:i] + text[j:]

    # A. §3 schemes paragraph: "When Phase-2 classifies a target text as
    # instantiating value-based practical reasoning..." through end of Table 6.
    text = cut(
        text,
        "When Phase-2 classifies a target text as instantiating",
        "\\label{tab:top-values} \n\\end{table}",
    )

    # B. §discussion subsection on the value vocabulary.
    text = cut(
        text,
        "\\subsection{What does the value vocabulary tell us about what we measure?}",
        "On this corpus, what the protocol measures is closer to procedural argumentative hygiene than to substantive moral judgment.",
    )

    # C. Appendix: Counting units for the value lexicon → end of the four
    # value-related subsections.  The four are: Counting units, Values per
    # cell, Canonical value pairings, Cells with no extracted value.
    text = cut(
        text,
        "\\subsection{Counting units for the value lexicon}",
        "\\label{tab:empty-extractions}\n\\end{table}",
    )

    # Tidy: collapse runs of >2 blank lines to exactly 2.
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")

    DST.write_text(text, encoding="utf-8")
    print(f"wrote {DST}")
    print(f"  source: {len(SRC.read_text(encoding='utf-8').splitlines())} lines")
    print(f"  derived: {len(text.splitlines())} lines")


if __name__ == "__main__":
    main()
