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

"""Unit tests for NCBI accession parsing and taxonomy XML parsing (no network)."""

from bionemo.evo2.utils import ncbi_taxonomy
from bionemo.evo2.utils.ncbi_taxonomy import (
    EntrezClientConfig,
    Evo2TaxonomyLineage,
    _taxon_lineage_from_xml,
    build_taxonomy_data_for_fasta_headers,
    parse_accession_from_fasta_header,
)


def test_parse_accession_ref_pipe() -> None:
    assert parse_accession_from_fasta_header(">ref|NC_045512.2|description") == "NC_045512.2"


def test_parse_accession_gb_pipe() -> None:
    assert parse_accession_from_fasta_header(">gb|AF123456.1|") == "AF123456.1"


def test_parse_accession_first_token_refseq() -> None:
    assert parse_accession_from_fasta_header(">NM_001301011.2 Homo sapiens") == "NM_001301011.2"


def test_parse_accession_first_token_legacy() -> None:
    assert parse_accession_from_fasta_header(">D00214.1 Bovine enterovirus genomic RNA") == "D00214.1"


def test_parse_accession_no_match() -> None:
    assert parse_accession_from_fasta_header(">contig_001 random") is None


def test_build_taxonomy_data_continues_on_error(monkeypatch) -> None:
    errors: list[tuple[str, str]] = []
    progress: list[tuple[int, int, int, int, str]] = []
    checkpoints: list[tuple[int, int, int, int, list[str]]] = []

    def fake_fetch(accession: str, cfg: EntrezClientConfig) -> Evo2TaxonomyLineage:
        del cfg
        if accession == "X12345.1":
            raise RuntimeError("boom")
        return Evo2TaxonomyLineage(species=accession)

    monkeypatch.setattr(ncbi_taxonomy, "fetch_lineage_for_accession", fake_fetch)

    out = build_taxonomy_data_for_fasta_headers(
        [
            ">NC_045512.2 SARS-CoV-2",
            ">X12345.1 legacy accession that fails",
            ">D00214.1 Bovine enterovirus genomic RNA",
        ],
        EntrezClientConfig(email="user@example.com"),
        continue_on_error=True,
        progress_every=1,
        progress_callback=lambda processed, total, successes, failures, accession: progress.append(
            (processed, total, successes, failures, accession)
        ),
        error_callback=lambda accession, error: errors.append((accession, str(error))),
        checkpoint_every=1,
        checkpoint_callback=lambda taxonomy_data, processed, total, successes, failures: checkpoints.append(
            (processed, total, successes, failures, sorted(taxonomy_data))
        ),
    )

    assert sorted(out) == ["D00214.1", "NC_045512.2"]
    assert errors == [("X12345.1", "boom")]
    assert progress[-1] == (3, 3, 2, 1, "D00214.1")
    assert checkpoints[-1] == (3, 3, 2, 1, ["D00214.1", "NC_045512.2"])


def test_taxon_lineage_from_xml_minimal() -> None:
    xml = b"""<?xml version="1.0"?>
<TaxaSet>
  <Taxon>
    <TaxId>9606</TaxId>
    <ScientificName>Homo sapiens</ScientificName>
    <Rank>species</Rank>
    <Division>Primates</Division>
    <LineageEx>
      <Taxon>
        <TaxId>2759</TaxId>
        <ScientificName>Eukaryota</ScientificName>
        <Rank>superkingdom</Rank>
      </Taxon>
      <Taxon>
        <TaxId>7711</TaxId>
        <ScientificName>Chordata</ScientificName>
        <Rank>phylum</Rank>
      </Taxon>
    </LineageEx>
  </Taxon>
</TaxaSet>
"""
    lin = _taxon_lineage_from_xml(xml)
    assert lin.domain == "Eukaryota"
    assert lin.phylum == "Chordata"
    assert lin.species == "Homo sapiens"


def test_taxon_lineage_from_xml_virus_lineage_string() -> None:
    """Virus: use semicolon Lineage tail + terminal ScientificName (no-rank species)."""
    xml = b"""<?xml version="1.0"?>
<TaxaSet>
  <Taxon>
    <TaxId>2697049</TaxId>
    <ScientificName>Severe acute respiratory syndrome coronavirus 2</ScientificName>
    <Rank>no rank</Rank>
    <Division>Viruses</Division>
    <Lineage>Viruses; Riboviria; Orthornavirae; Pisuviricota; Pisoniviricetes; Nidovirales; Cornidovirineae; Coronaviridae; Orthocoronavirinae; Betacoronavirus; Sarbecovirus; Betacoronavirus pandemicum</Lineage>
    <LineageEx>
      <Taxon>
        <TaxId>10239</TaxId>
        <ScientificName>Viruses</ScientificName>
        <Rank>superkingdom</Rank>
      </Taxon>
    </LineageEx>
  </Taxon>
</TaxaSet>
"""
    lin = _taxon_lineage_from_xml(xml)
    # Last seven ``Lineage`` segments fill d,p,c,o,f,g,s in order (ICTV path, not literal ranks).
    assert lin.domain == "Nidovirales"
    assert lin.clazz == "Coronaviridae"
    assert lin.order == "Orthocoronavirinae"
    assert lin.species == "Severe acute respiratory syndrome coronavirus 2"
