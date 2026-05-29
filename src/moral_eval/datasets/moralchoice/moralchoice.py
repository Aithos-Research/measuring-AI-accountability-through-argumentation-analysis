# src/moral_eval/datasets/moralchoice/moralchoice.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from huggingface_hub import hf_hub_download
from inspect_ai.dataset import Sample, MemoryDataset


class MoralChoice:
    """
    MoralChoice dataset wrapper (merged CSV version).

    Reads merged rows from HF:
      - derived/moralchoice_low_merged.csv
      - derived/moralchoice_high_merged.csv

    CSV columns expected (subset is fine, but these are used):
      scenario_id, ambiguity, generation_type, generation_rule,
      context, action1, action2, values_to_do, values_not_to_do,
      (plus any other columns you want to keep in metadata)

    Public API:
      - build_merged(record_to_sample=..., limit=...) -> MemoryDataset
      - build(limit=...) remains available (legacy), but merged runs should use build_merged().
    """

    # ---------------------------------------------------------------------
    # HF-hosted merged CSVs — keep these paths exactly (per your request)
    # ---------------------------------------------------------------------
    HF_REPO_ID = "Aithos/moralchoice"
    HF_LOW_MERGED_PATH = "derived/moralchoice_low_merged.csv"
    HF_HIGH_MERGED_PATH = "derived/moralchoice_high_merged.csv"
    HF_REVISION = None  # optionally pin commit hash

    def __init__(
        self,
        *,
        ambiguity: str = "low",
        template: str = "ab",  # accepted for compatibility; not used by build_merged
        include_system_instruction: bool = True,  # accepted for compatibility
        include_both_orderings: bool = True,  # accepted for compatibility
        split: str = "train",  # accepted for compatibility
    ):
        if ambiguity not in ("low", "high"):
            raise ValueError("ambiguity must be 'low' or 'high'")
        self.ambiguity = ambiguity

        # keep args for backwards-compat / existing callsites
        self.template = template
        self.include_system_instruction = include_system_instruction
        self.include_both_orderings = include_both_orderings
        self.split = split

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_list_cell(x: Any) -> list[str]:
        """Parse JSON list cells like '["Helpful", "Responsible"]' into list[str]."""
        if x is None:
            return []
        if isinstance(x, list):
            return [str(v).strip() for v in x if str(v).strip()]
        s = str(x).strip()
        if not s or s == "[]":
            return []
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(v).strip() for v in arr if str(v).strip()]
        except Exception:
            return []
        return []

    @classmethod
    def _download_hf_csv(cls, repo_id: str, repo_path: str, *, revision: str | None = None) -> Path:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=repo_path,
            revision=revision,
        )
        return Path(local_path)

    def _load_merged_rows(self) -> list[dict]:
        repo_path = self.HF_LOW_MERGED_PATH if self.ambiguity == "low" else self.HF_HIGH_MERGED_PATH
        csv_path = self._download_hf_csv(self.HF_REPO_ID, repo_path, revision=self.HF_REVISION)

        df = pd.read_csv(csv_path)

        required = ["scenario_id", "context", "action1", "action2", "values_to_do", "values_not_to_do"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Merged CSV missing required columns: {missing} ({csv_path})")

        rows: list[dict] = []
        for _, r in df.iterrows():
            row = dict(r)

            # normalize core fields
            row["scenario_id"] = str(row.get("scenario_id", "")).strip()
            row["ambiguity"] = str(row.get("ambiguity", self.ambiguity)).strip() or self.ambiguity
            row["context"] = "" if row.get("context") is None else str(row.get("context"))
            row["action1"] = "" if row.get("action1") is None else str(row.get("action1"))
            row["action2"] = "" if row.get("action2") is None else str(row.get("action2"))

            # parse value columns
            row["values_to_do"] = self._parse_json_list_cell(row.get("values_to_do"))
            row["values_not_to_do"] = self._parse_json_list_cell(row.get("values_not_to_do"))

            if not row["scenario_id"]:
                continue

            # drop unusable rows
            if not row["values_to_do"] or not row["values_not_to_do"]:
                continue

            rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Public builder: merged CSV -> MemoryDataset
    # ------------------------------------------------------------------

    def build_merged(
        self,
        *,
        record_to_sample: Callable[[dict], Any] | None = None,
        limit: int | None = None,
    ) -> MemoryDataset:
        """
        Build MemoryDataset from the merged CSV.

        - If record_to_sample returns a Sample -> used directly.
        - If record_to_sample returns a dict -> wrapped into Sample(metadata=dict).
        - If record_to_sample is None -> a default Sample is created.
        """
        rows = self._load_merged_rows()
        if limit is not None:
            rows = rows[:limit]

        samples: list[Sample] = []

        for row in rows:
            # default “topic” fields for your scripts (since merged CSV doesn't have topic/topic_group)
            # You can change this mapping, but this keeps your existing logging/export columns happy.
            row.setdefault("topic", row.get("generation_rule", ""))
            row.setdefault("topic_group", row.get("generation_type", ""))

            # also keep “dilemma_idx” alias for old scripts if any remain
            row.setdefault("dilemma_idx", row["scenario_id"])

            if record_to_sample is None:
                # minimal default
                s = Sample(
                    id=row["scenario_id"],
                    input=row["context"],
                    target=" ",
                    metadata=row,
                )
                samples.append(s)
                continue

            obj = record_to_sample(row)

            if isinstance(obj, Sample):
                # ensure id is set
                if not getattr(obj, "id", None):
                    obj.id = row["scenario_id"]  # type: ignore[attr-defined]
                # ensure metadata at least contains row
                if not getattr(obj, "metadata", None):
                    obj.metadata = {}  # type: ignore[attr-defined]
                obj.metadata = {**row, **(obj.metadata or {})}  # type: ignore[attr-defined]
                samples.append(obj)
            elif isinstance(obj, dict):
                # wrap dict in Sample so downstream code can do r.metadata["scenario_id"]
                samples.append(
                    Sample(
                        id=str(obj.get("scenario_id", row["scenario_id"])),
                        input=str(obj.get("input", "")),
                        target=str(obj.get("target", " ")),
                        metadata=obj,
                    )
                )
            else:
                raise TypeError(
                    "record_to_sample must return either inspect_ai.dataset.Sample or dict"
                )

        return MemoryDataset(samples)

    # ------------------------------------------------------------------
    # Legacy build() retained (so old imports won't break)
    # ------------------------------------------------------------------
    def build(self, *, limit: int | None = None) -> MemoryDataset:
        """
        Legacy shim: for your new pipeline, prefer build_merged().
        """
        return self.build_merged(limit=limit)
