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

"""Resolve NCBI nucleotide accessions to Evo2 taxonomy lineages via Entrez (E-utilities).

NCBI requires an email contact for E-utilities; optionally use an API key for higher rate limits.
See https://www.ncbi.nlm.nih.gov/books/NBK25497/

**Viruses:** For ``Division == Viruses``, the NCBI ``Lineage`` string (semicolon-separated) is
split and the **last seven** segments populate ``r__…`` through ``s__…`` in order (see
``_construct_taxonomy_token``), which tracks deep ICTV-style paths (suborders, subfamilies, etc.).
The terminal taxon ``ScientificName`` overrides ``species`` when the record uses ICTV names
(often ``Rank`` is ``no rank``).
"""

from __future__ import annotations

from collections.abc import Callable
import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


# Defined locally (instead of imported from ``bionemo.evo2.utils.config``) so
# this module doesn't pull in NeMo/Megatron at import time, which lets the
# taxonomy YAML be generated outside the bionemo-framework container. The
# schema mirrors ``config.Evo2TaxonomyLineage`` exactly.
class Evo2TaxonomyLineage(BaseModel):
    """Pydantic model class that defines the source lineage of a DNA sequence."""

    domain: None | str = None
    phylum: None | str = None
    clazz: None | str = None
    order: None | str = None
    family: None | str = None
    genus: None | str = None
    species: None | str = None


_ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# RefSeq / INSDC style accessions (first token of header).
_ACCESSION_PATTERN = re.compile(
    r"^(?:[A-Z]{1,2}_\d+(?:\.\d+)?|[A-Z]\d{5,6}(?:\.\d+)?|[A-Z]{2}\d{6}(?:\.\d+)?|[A-Z]{4}\d{8,}(?:\.\d+)?)$"
)


@dataclass(frozen=True)
class EntrezClientConfig:
    """Parameters for NCBI Entrez HTTP calls."""

    email: str
    api_key: str | None = None
    delay_s: float = 0.35
    timeout_s: float = 120.0
    max_retries: int = 3


ProgressCallback = Callable[[int, int, int, int, str], None]
ErrorCallback = Callable[[str, RuntimeError], None]
CheckpointCallback = Callable[[dict[str, Evo2TaxonomyLineage], int, int, int, int], None]


def read_fasta_headers(path: Path | str) -> list[str]:
    """Return header description lines (without ``>``) for each FASTA record."""
    headers: list[str] = []
    p = Path(path)
    with p.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                headers.append(line[1:].strip())
    return headers


def parse_accession_from_fasta_header(header: str) -> str | None:
    """Extract a nucleotide accession from a FASTA header line.

    Supports common NCBI ``ref|ACCESSION|`` / ``gb|ACCESSION|`` patterns and RefSeq-style first tokens.

    Args:
        header: Header text with or without a leading ``>``.

    Returns:
        Accession string, or None if no pattern matched.
    """
    line = header.strip()
    if line.startswith(">"):
        line = line[1:]
    line = line.strip()

    pipe = re.search(r"(?:ref|gb|emb|dbj|pdb)\|([^|]+)\|", line, re.IGNORECASE)
    if pipe:
        return pipe.group(1).strip()

    first = line.split()[0] if line else ""
    if _ACCESSION_PATTERN.match(first):
        return first

    return None


def _entrez_params(cfg: EntrezClientConfig, extra: dict[str, str]) -> str:
    params = {"tool": "bionemo_evo2_taxonomy", "email": cfg.email, **extra}
    if cfg.api_key:
        params["api_key"] = cfg.api_key
    return urllib.parse.urlencode(params)


def _http_get(url: str, cfg: EntrezClientConfig) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": f"bionemo-evo2-taxonomy/1.0 ({cfg.email})"},
        method="GET",
    )
    retries = max(cfg.max_retries, 1)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"NCBI HTTP {e.code}: {e.reason}") from e
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as e:
            if attempt == retries:
                raise RuntimeError(f"NCBI request failed: {e}") from e
            time.sleep(min(cfg.delay_s * attempt, 5.0))

    raise RuntimeError("NCBI request failed after exhausting retries.")


def _esearch_db(cfg: EntrezClientConfig, db: str, term: str) -> list[str]:
    """Run esearch and return id list."""
    q = _entrez_params(
        cfg,
        {"db": db, "term": term, "retmode": "json", "retmax": "5"},
    )
    url = f"{_ENTREZ_BASE}/esearch.fcgi?{q}"
    time.sleep(cfg.delay_s)
    raw = _http_get(url, cfg)
    data = json.loads(raw.decode())
    res = data.get("esearchresult") or {}
    return list(res.get("idlist") or [])


def _esummary_taxid_nuccore(cfg: EntrezClientConfig, nucl_id: str) -> str | None:
    q = _entrez_params(cfg, {"db": "nuccore", "id": nucl_id, "retmode": "json"})
    url = f"{_ENTREZ_BASE}/esummary.fcgi?{q}"
    time.sleep(cfg.delay_s)
    raw = _http_get(url, cfg)
    data = json.loads(raw.decode())
    result = data.get("result") or {}
    uid = (result.get("uids") or [None])[0]
    if uid is None:
        return None
    rec = result.get(uid) or {}
    tid = rec.get("taxid")
    if tid is None:
        return None
    return str(tid)


def _efetch_taxonomy_xml(cfg: EntrezClientConfig, taxid: str) -> bytes:
    q = _entrez_params(cfg, {"db": "taxonomy", "id": taxid, "retmode": "xml"})
    url = f"{_ENTREZ_BASE}/efetch.fcgi?{q}"
    time.sleep(cfg.delay_s)
    return _http_get(url, cfg)


def _assign_rank(name: str, rank: str, slots: dict[str, str | None]) -> None:
    rank = rank.strip().lower()
    if not name:
        return
    if rank == "superkingdom":
        slots["domain"] = name
    elif rank == "realm" and slots.get("domain") is None:
        # Viral lineages (e.g. Riboviria) often use realm before kingdom.
        slots["domain"] = name
    elif rank == "kingdom" and slots.get("domain") is None:
        slots["domain"] = name
    elif rank == "phylum":
        slots["phylum"] = name
    elif rank == "class":
        slots["clazz"] = name
    elif rank == "order":
        slots["order"] = name
    elif rank == "suborder" and slots.get("order") is None:
        slots["order"] = name
    elif rank == "family":
        slots["family"] = name
    elif rank == "subfamily" and slots.get("family") is None:
        slots["family"] = name
    elif rank == "genus":
        slots["genus"] = name
    elif rank == "subgenus" and slots.get("genus") is None:
        slots["genus"] = name
    elif rank == "species":
        slots["species"] = name


def _lineage_string_to_slots(lineage_text: str) -> dict[str, str | None]:
    """Map NCBI semicolon-separated Lineage to seven Evo2 slots.

    Viral taxonomies often use suborder/subfamily/subgenus and many ``no rank`` nodes; the
    full ``Lineage`` string preserves order. We take the **last seven** segments so the
    deepest levels (typically order→species) align with ``r__…;k__…;c__…;…;s__…`` prompting.
    """
    parts = [p.strip() for p in lineage_text.split(";") if p.strip()]
    slots = {
        "domain": None,
        "phylum": None,
        "clazz": None,
        "order": None,
        "family": None,
        "genus": None,
        "species": None,
    }
    keys = ("domain", "phylum", "clazz", "order", "family", "genus", "species")
    if not parts:
        return slots
    if len(parts) >= 7:
        tail = parts[-7:]
    else:
        tail = parts
    for i, key in enumerate(keys[: len(tail)]):
        slots[key] = tail[i]
    return slots


def _taxon_lineage_from_xml(xml_bytes: bytes) -> Evo2TaxonomyLineage:
    """Parse taxonomy efetch XML into Evo2TaxonomyLineage.

    Prefer the NCBI ``Lineage`` string when present (strong for viruses). Otherwise use
    ``LineageEx`` with rank-based mapping, including realm/suborder/subfamily for viruses.
    """
    root = ET.fromstring(xml_bytes)
    slots: dict[str, str | None] = {
        "domain": None,
        "phylum": None,
        "clazz": None,
        "order": None,
        "family": None,
        "genus": None,
        "species": None,
    }

    main = root.find(".//TaxaSet/Taxon")
    lineage_text = (main.findtext("Lineage") or "").strip() if main is not None else ""

    # Viral taxonomies use many ``no rank`` and sub-* ranks; the semicolon ``Lineage`` string
    # matches Evo2's seven slots well via the last seven segments. Cellular lineages are longer
    # and rank-based LineageEx mapping is safer for bacteria/eukaryotes.
    is_virus = False
    if main is not None:
        div = (main.findtext("Division") or "").strip()
        is_virus = div == "Viruses" or lineage_text.startswith("Viruses")

    if lineage_text and is_virus:
        slots = _lineage_string_to_slots(lineage_text)
    else:
        lineage_ex = root.find(".//LineageEx")
        if lineage_ex is not None:
            for taxon in lineage_ex.findall("Taxon"):
                rank = taxon.findtext("Rank") or ""
                name = (taxon.findtext("ScientificName") or "").strip()
                _assign_rank(name, rank, slots)

    if main is not None:
        rank = (main.findtext("Rank") or "").strip().lower()
        name = (main.findtext("ScientificName") or "").strip()
        # Viral species often have Rank ``no rank`` on the terminal taxon; still use its name.
        if name and rank in ("species", "no rank", "strain"):
            slots["species"] = name
        elif rank == "genus" and name and slots["genus"] is None:
            slots["genus"] = name

    return Evo2TaxonomyLineage(
        domain=slots["domain"],
        phylum=slots["phylum"],
        clazz=slots["clazz"],
        order=slots["order"],
        family=slots["family"],
        genus=slots["genus"],
        species=slots["species"],
    )


def fetch_lineage_for_accession(accession: str, cfg: EntrezClientConfig) -> Evo2TaxonomyLineage:
    """Resolve a nucleotide accession to taxonomy using NCBI Entrez.

    Args:
        accession: Nucleotide accession (e.g. ``NC_045512.2``).
        cfg: Entrez client options (email required by NCBI).

    Returns:
        Parsed lineage for Evo2 prompting.

    Raises:
        RuntimeError: If the accession cannot be resolved or taxonomy is missing.
    """
    term = f"{accession}[Accession]"
    ids = _esearch_db(cfg, "nuccore", term)
    if not ids:
        ids = _esearch_db(cfg, "nuccore", accession)

    if not ids:
        raise RuntimeError(f"No NCBI nuccore record found for accession {accession!r}.")

    taxid = _esummary_taxid_nuccore(cfg, ids[0])
    if not taxid:
        raise RuntimeError(f"No taxid on nuccore summary for accession {accession!r}.")

    xml = _efetch_taxonomy_xml(cfg, taxid)
    return _taxon_lineage_from_xml(xml)


def build_taxonomy_data_for_fasta_headers(
    headers: list[str],
    cfg: EntrezClientConfig,
    *,
    key_from_accession: bool = True,
    existing_taxonomy_data: dict[str, Evo2TaxonomyLineage] | None = None,
    skip_accessions: set[str] | None = None,
    continue_on_error: bool = False,
    progress_every: int | None = None,
    progress_callback: ProgressCallback | None = None,
    error_callback: ErrorCallback | None = None,
    checkpoint_every: int | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> dict[str, Evo2TaxonomyLineage]:
    """Build a ``taxonomy_data`` map from FASTA headers.

    Keys are chosen so that ``key in seqid`` matches during preprocessing (substring match).
    By default the key is the accession string, which should appear in the header.

    Args:
        headers: FASTA header lines (with or without ``>``).
        cfg: Entrez configuration.
        key_from_accession: If True, map key is the parsed accession; otherwise use full seqid (first token).
        existing_taxonomy_data: Existing results to preserve and extend.
        skip_accessions: Accessions that are already complete and should not be re-fetched.
        continue_on_error: If True, log/callback per-accession failures and continue processing.
        progress_every: Emit progress every N processed unique accessions when paired with ``progress_callback``.
        progress_callback: Invoked as ``(processed, total, successes, failures, accession)``.
        error_callback: Invoked for per-accession ``RuntimeError`` exceptions.
        checkpoint_every: Write/emit checkpoints every N successful accessions when paired with
            ``checkpoint_callback``.
        checkpoint_callback: Invoked as ``(out, processed, total, successes, failures)``.

    Returns:
        Mapping suitable for :class:`Evo2PreprocessingConfig`.``taxonomy_data``.
    """
    out: dict[str, Evo2TaxonomyLineage] = dict(existing_taxonomy_data or {})
    seen_acc: set[str] = set(skip_accessions or ())
    entries: list[tuple[str, str]] = []

    for header in headers:
        acc = parse_accession_from_fasta_header(header)
        if not acc:
            continue
        if acc in seen_acc:
            continue
        seen_acc.add(acc)
        key = acc if key_from_accession else header.split()[0].lstrip(">")
        entries.append((acc, key))

    total = len(entries)
    failures = 0

    for processed, (acc, key) in enumerate(entries, start=1):
        try:
            lineage = fetch_lineage_for_accession(acc, cfg)
        except RuntimeError as e:
            failures += 1
            if error_callback is not None:
                error_callback(acc, e)
            if not continue_on_error:
                raise
        else:
            out[key] = lineage
            if checkpoint_callback is not None and checkpoint_every and len(out) % checkpoint_every == 0:
                checkpoint_callback(out, processed, total, len(out), failures)

        if progress_callback is not None and progress_every and (processed % progress_every == 0 or processed == total):
            progress_callback(processed, total, len(out), failures, acc)

    if checkpoint_callback is not None and checkpoint_every and out and len(out) % checkpoint_every != 0:
        checkpoint_callback(out, total, total, len(out), failures)

    return out
