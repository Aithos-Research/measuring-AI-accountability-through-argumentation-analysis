"""Re-export samples.json for an existing deliberative_justification_v2 run.

Use this after fixing a bug in the exporter to refresh samples.json from the
already-stored per-model eval logs -- no API calls are issued, the cached
``judge1_raw_*`` / ``judge2_raw_*`` text is just re-parsed by the current
``export_samples_to_template`` implementation.

Usage (from project root, in the venv)::

    .\\.venv\\Scripts\\python.exe scripts/reexport_samples.py \\
        --log-dir logs/deliberative_justification_v2/20260515_095530/cot \\
        --phase2 claude_sonnet_4_6 \\
        --phase4 gpt_5_5 --phase4 claude_sonnet_4_6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-dir", required=True, type=Path,
                   help="Directory containing the per-model .json eval logs.")
    p.add_argument("--phase2", action="append", required=True,
                   help="Phase-2 judge alias as it appears in metadata keys "
                        "(e.g. claude_sonnet_4_6).  Repeatable.")
    p.add_argument("--phase4", action="append", required=True,
                   help="Phase-4 judge alias.  Repeatable "
                        "(e.g. --phase4 gpt_5_5 --phase4 claude_sonnet_4_6).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path.  Default: <log-dir>/samples.json (overwrites).")
    args = p.parse_args(argv)

    if not args.log_dir.exists():
        raise SystemExit(f"log dir not found: {args.log_dir}")

    out = args.out or (args.log_dir / "samples.json")

    # Import after sys.path setup.  Importing the module triggers the task
    # registration code, which requires task config to be resolvable; point at
    # the smoke YAML so it loads cleanly.
    import os
    os.environ.setdefault(
        "MORAL_EVAL_EXPERIMENTS_FILE",
        str(PROJECT_ROOT / "src" / "config" / "test_walton_moralchoice.yaml"),
    )
    from moral_eval.tasks.explainability.deliberative_justification_v2 import (
        export_samples_to_template,
    )

    written = export_samples_to_template(
        args.log_dir, out, args.phase2, args.phase4,
    )
    print(f"re-exported {written} records to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
