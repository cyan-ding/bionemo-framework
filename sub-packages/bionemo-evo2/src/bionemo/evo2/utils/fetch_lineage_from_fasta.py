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
from nemo.utils import logging

from bionemo.evo2.utils.ncbi_taxonomy import (
    EntrezClientConfig,
    build_taxonomy_data_for_fasta_headers,
    parse_accession_from_fasta_header,
    read_fasta_headers,
)


def _lineage_to_yaml_dict(lineage) -> dict:
    d = lineage.model_dump()
    return {k: v for k, v in d.items() if v is not None}


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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    headers = read_fasta_headers(args.fasta)
    if not headers:
        logging.error("No FASTA records found in %s", args.fasta)
        sys.exit(1)

    missing = [h for h in headers if parse_accession_from_fasta_header(h) is None]
    for h in missing:
        logging.warning("Could not parse nucleotide accession from header (skipped): %s", h[:200])

    cfg = EntrezClientConfig(email=args.email, api_key=args.api_key, delay_s=args.delay)
    try:
        taxonomy_data = build_taxonomy_data_for_fasta_headers(headers, cfg)
    except RuntimeError as e:
        logging.error("%s", e)
        sys.exit(1)

    if not taxonomy_data:
        logging.error("No taxonomy entries produced (check accessions and network).")
        sys.exit(1)

    payload = {
        "taxonomy_data": {k: _lineage_to_yaml_dict(v) for k, v in taxonomy_data.items()},
    }
    text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)

    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
        logging.info("Wrote %s (%d accessions)", args.output, len(taxonomy_data))
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
