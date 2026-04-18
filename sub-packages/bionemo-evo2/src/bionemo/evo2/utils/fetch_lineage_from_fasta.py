# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI: build ``taxonomy_data`` for Evo2 preprocessing from NCBI nucleotide accessions in FASTA headers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from bionemo.evo2.utils.ncbi_taxonomy import (
    EntrezClientConfig,
    Evo2TaxonomyLineage,
    build_taxonomy_data_for_fasta_headers,
    parse_accession_from_fasta_header,
    read_fasta_headers,
)


def _lineage_to_yaml_dict(lineage) -> dict:
    d = lineage.model_dump()
    return {k: v for k, v in d.items() if v is not None}


def _write_taxonomy_yaml(path: Path, taxonomy_data: dict[str, object]) -> None:
    payload = {
        "taxonomy_data": {k: _lineage_to_yaml_dict(v) for k, v in taxonomy_data.items()},
    }
    text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _read_existing_taxonomy_yaml(path: Path) -> dict[str, Evo2TaxonomyLineage]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("taxonomy_data") or {}
    if not isinstance(entries, dict):
        raise RuntimeError(f"Existing YAML at {path} does not contain a taxonomy_data mapping.")
    out: dict[str, Evo2TaxonomyLineage] = {}
    for key, value in entries.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        out[key] = Evo2TaxonomyLineage.model_validate(value)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read FASTA headers, resolve NCBI nucleotide accessions to taxonomy lineages via Entrez, "
            "and write a YAML fragment for Evo2PreprocessingConfig.taxonomy_data. "
            "NCBI requires --email; use --api-key for higher rate limits."
        )
    )
    p.add_argument("--fasta", type=Path, required=True, help="Path to a FASTA file.")
    p.add_argument(
        "--email",
        type=str,
        required=True,
        help="Contact email for NCBI E-utilities (required by NCBI policy).",
    )
    p.add_argument("--api-key", type=str, default=None, help="Optional NCBI API key.")
    p.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Seconds to sleep between Entrez requests (default: 0.35).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write YAML here (default: stdout).",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N unique accessions processed (default: 100). Use 0 to disable.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help=(
            "Write partial YAML every N successful accessions when --output is set "
            "(default: 100). Use 0 to disable."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    headers = read_fasta_headers(args.fasta)
    if not headers:
        print(f"No FASTA records found in {args.fasta}", file=sys.stderr)
        sys.exit(1)

    missing = [h for h in headers if parse_accession_from_fasta_header(h) is None]
    for h in missing:
        print(
            f"Could not parse nucleotide accession from header (skipped): {h[:200]}",
            file=sys.stderr,
        )

    existing_taxonomy_data: dict[str, Evo2TaxonomyLineage] = {}
    completed_accessions: set[str] = set()
    if args.output is not None and args.output.exists():
        existing_taxonomy_data = _read_existing_taxonomy_yaml(args.output)
        completed_accessions = set(existing_taxonomy_data)
        print(
            f"Loaded {len(existing_taxonomy_data)} existing taxonomy entries from {args.output}; "
            "will skip them on resume.",
            file=sys.stderr,
        )

    cfg = EntrezClientConfig(email=args.email, api_key=args.api_key, delay_s=args.delay)
    progress_every = args.progress_every if args.progress_every > 0 else None
    checkpoint_every = args.checkpoint_every if args.checkpoint_every > 0 else None

    if checkpoint_every and args.output is None:
        print("Checkpointing disabled because --output was not provided.", file=sys.stderr)
        checkpoint_every = None

    def on_progress(processed: int, total: int, successes: int, failures: int, accession: str) -> None:
        print(
            f"Processed {processed}/{total} unique accessions "
            f"(ok={successes}, failed={failures}, last={accession})",
            file=sys.stderr,
        )

    def on_error(accession: str, error: RuntimeError) -> None:
        print(f"Failed to resolve accession {accession!r}: {error}", file=sys.stderr)

    def on_checkpoint(
        taxonomy_data: dict[str, object],
        processed: int,
        total: int,
        successes: int,
        failures: int,
    ) -> None:
        if args.output is None:
            return
        _write_taxonomy_yaml(args.output, taxonomy_data)
        print(
            f"Checkpointed {args.output} ({successes} accessions written; "
            f"processed {processed}/{total}, failed={failures})",
            file=sys.stderr,
        )

    taxonomy_data = build_taxonomy_data_for_fasta_headers(
        headers,
        cfg,
        existing_taxonomy_data=existing_taxonomy_data,
        skip_accessions=completed_accessions,
        continue_on_error=True,
        progress_every=progress_every,
        progress_callback=on_progress,
        error_callback=on_error,
        checkpoint_every=checkpoint_every,
        checkpoint_callback=on_checkpoint,
    )

    if not taxonomy_data:
        print("No taxonomy entries produced (check accessions and network).", file=sys.stderr)
        sys.exit(1)

    if args.output is not None:
        _write_taxonomy_yaml(args.output, taxonomy_data)
        print(f"Wrote {args.output} ({len(taxonomy_data)} accessions)", file=sys.stderr)
    else:
        payload = {
            "taxonomy_data": {k: _lineage_to_yaml_dict(v) for k, v in taxonomy_data.items()},
        }
        sys.stdout.write(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))


if __name__ == "__main__":
    main()
