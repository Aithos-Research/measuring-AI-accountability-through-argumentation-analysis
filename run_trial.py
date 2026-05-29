"""Local launcher for the deliberative_justification_v2 trial.

Run with the project's virtual-env Python so the right packages are loaded:

    .\.venv\Scripts\python.exe run_trial.py                            # default: src/config/experiments.yaml
    .\.venv\Scripts\python.exe run_trial.py --test                     # small generic smoke test (src/config/test_run.yaml)
    .\.venv\Scripts\python.exe run_trial.py --dataset daily            # Walton paper: 100 DailyDilemmas
    .\.venv\Scripts\python.exe run_trial.py --dataset airisk           # Walton paper: 100 AIRiskDilemmas
    .\.venv\Scripts\python.exe run_trial.py --dataset moralchoice      # Walton paper: 100 MoralChoice
    .\.venv\Scripts\python.exe run_trial.py --test --dataset daily     # 5-sample smoke variant of the daily paper config
    .\.venv\Scripts\python.exe run_trial.py --test --dataset airisk    # 5-sample smoke variant of the airisk paper config
    .\.venv\Scripts\python.exe run_trial.py --test --dataset moralchoice  # 5-sample smoke variant of the moralchoice paper config

The wrapper handles three things you'd otherwise have to set up by hand:

1. Switches `requests` to the Windows OS trust store via `truststore`, so HuggingFace
   SSL works without manual CA-bundle juggling.
2. Loads .env and aliases the project's legacy `HUGGINGFACE_ACCESS_KEY` onto the
   standard `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` names that huggingface_hub looks for.
3. Selects the right experiments YAML based on --test and/or --dataset flags.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_CFG_DIR = _HERE / "src" / "config"
_DATASET_TO_CONFIG = {
    "daily": _CFG_DIR / "experiments_walton_daily.yaml",
    "airisk": _CFG_DIR / "experiments_walton_airisk.yaml",
    "moralchoice": _CFG_DIR / "experiments_walton_moralchoice.yaml",
}
_DATASET_TO_TEST_CONFIG = {
    "daily": _CFG_DIR / "test_walton_daily.yaml",
    "airisk": _CFG_DIR / "test_walton_airisk.yaml",
    "moralchoice": _CFG_DIR / "test_walton_moralchoice.yaml",
}


def _switch_to_test_config() -> None:
    os.environ["MORAL_EVAL_EXPERIMENTS_FILE"] = str(_CFG_DIR / "test_run.yaml")


def _switch_to_dataset_config(dataset_key: str, *, test: bool = False) -> None:
    mapping = _DATASET_TO_TEST_CONFIG if test else _DATASET_TO_CONFIG
    path = mapping[dataset_key]
    if not path.exists():
        raise FileNotFoundError(f"Config for --dataset {dataset_key} not found at {path}")
    os.environ["MORAL_EVAL_EXPERIMENTS_FILE"] = str(path)


def _bootstrap_ssl_and_env() -> None:
    # Force UTF-8 + suppress Inspect-AI's rich-traceback console.
    #
    # Background: Inspect-AI's error handler renders a coloured traceback via
    # rich which emits "→" markers.  On a cp1252 Windows console that write
    # raises UnicodeEncodeError, which then masks the real exception AND
    # kills the whole eval task (every sample interrupted, 0/200 completed).
    #
    # Three layers of fix, applied in order:
    #   1.  PYTHONUTF8=1 and PYTHONIOENCODING=utf-8 env vars.
    #   2.  Replace sys.stdout / sys.stderr with fresh TextIOWrappers built
    #       on the underlying byte buffer with encoding=utf-8 errors=replace.
    #       This survives even if a library kept a reference to the old
    #       wrapper because the new wrapper IS the new sys.stdout.
    #   3.  Set INSPECT_DISPLAY=plain so inspect-ai uses a plain-text
    #       progress display instead of rich, avoiding the issue entirely.
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["INSPECT_DISPLAY"] = "plain"

    import io as _io
    for stream_name in ("stdout", "stderr"):
        s = getattr(sys, stream_name, None)
        if s is None:
            continue
        buf = getattr(s, "buffer", None)
        if buf is None:
            continue
        try:
            new_stream = _io.TextIOWrapper(
                buf, encoding="utf-8", errors="replace",
                newline="\n", line_buffering=True,
            )
            setattr(sys, stream_name, new_stream)
        except Exception:
            # Last-ditch fallback: reconfigure the existing wrapper.
            try:
                s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
            except Exception:
                pass

    import truststore

    truststore.inject_into_ssl()

    from dotenv import load_dotenv

    load_dotenv(override=True)

    hf_key = os.environ.get("HUGGINGFACE_ACCESS_KEY") or os.environ.get("HF_TOKEN")
    if hf_key:
        os.environ["HF_TOKEN"] = hf_key
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use src/config/test_run.yaml (small smoke test).",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(_DATASET_TO_CONFIG),
        help=(
            "Run the Walton-paper config for one of the three datasets. "
            "Each variant points at data/selections/<dataset>_seed42_n100.json."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=("forced_choice", "open"),
        help=(
            "Walton prompt variant: `forced_choice` (A/B dilemmas) or `open` "
            "(free-form resolutions). Overrides the YAML `variant` field via "
            "the MORAL_EVAL_WALTON_VARIANT env var."
        ),
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "Route the seven reasoning-capable panel models through "
            "Inspect-AI's reasoning-model path instead of the CoT path. "
            "Defaults to the n=200 production config "
            "(src/config/experiments_walton_thinking.yaml); pair with "
            "--test for the 5-sample smoke version "
            "(src/config/test_walton_thinking.yaml)."
        ),
    )
    args = parser.parse_args()

    if args.thinking:
        thinking_cfg = "test_walton_thinking.yaml" if args.test else "experiments_walton_thinking.yaml"
        os.environ["MORAL_EVAL_EXPERIMENTS_FILE"] = str(_CFG_DIR / thinking_cfg)
    elif args.test and args.dataset:
        # --test --dataset X => 5-sample smoke variant of the chosen dataset.
        _switch_to_dataset_config(args.dataset, test=True)
    elif args.test:
        _switch_to_test_config()
    elif args.dataset:
        _switch_to_dataset_config(args.dataset)

    if args.variant:
        os.environ["MORAL_EVAL_WALTON_VARIANT"] = args.variant

    _bootstrap_ssl_and_env()

    from moral_eval.tasks.explainability.deliberative_justification_v2 import run

    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
