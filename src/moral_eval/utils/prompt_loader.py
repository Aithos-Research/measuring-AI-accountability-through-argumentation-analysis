from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parents[2] / "config" / "prompts"
WALTON_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "data" / "walton_schemes"


def load_prompt(name: str) -> str:
    """Load a prompt template from src/config/prompts."""
    path = PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt '{name}' not found at {path}")
    return path.read_text(encoding="utf-8")


WALTON_VARIANTS = ("forced_choice", "open")


def load_walton_prompt(name: str, variant: str | None = None) -> str:
    """Load a Walton-protocol prompt from data/walton_schemes/.

    The Walton prompts live under data/walton_schemes/ as the canonical
    upstream-managed source so they can be updated from a pull without
    competing with edits under src/config/prompts/.  The directory is now
    organised by **variant** (May 2026 revision):

    - ``forced_choice/`` -- A/B forced-choice dilemmas (original protocol).
    - ``open/`` -- free-form ``<answer>...</answer>`` resolutions, no A/B.

    Pass ``variant`` to select the subfolder.  ``variant=None`` reads the
    legacy flat layout (kept for back-compat where no subfolder exists).
    """
    if variant is None:
        path = WALTON_PROMPTS_DIR / name
    else:
        if variant not in WALTON_VARIANTS:
            raise ValueError(
                f"Unknown Walton variant '{variant}'. Valid: {WALTON_VARIANTS}"
            )
        path = WALTON_PROMPTS_DIR / variant / name
    if not path.exists():
        raise FileNotFoundError(f"Walton prompt '{name}' not found at {path}")
    return path.read_text(encoding="utf-8")


__all__ = [
    "load_prompt",
    "load_walton_prompt",
    "PROMPT_DIR",
    "WALTON_PROMPTS_DIR",
    "WALTON_VARIANTS",
]
