"""
CMI-PB Data Pipeline — fetch multi-omics immunology data and build a block tensor.

Fetches 7 omics assays from the CMI-PB REST API (v5.1), identifies complete
cases across all assays and requested timepoints, and assembles a tensor of
shape (m, p, n) with block partition vector b.

Usage
-----
    cd TBI
    conda activate claude
    python cmipb_pipeline.py          # fetch + build + report
    python cmipb_pipeline.py --days 0 1 3 14
"""

import json
import os
import urllib.request
import urllib.parse
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "https://www.cmi-pb.org/api/v5_1"
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CACHE_DIR = os.path.join(_REPO_ROOT, "data", "cmipb")

# Assay definitions: (table_name, variable_id_fields, value_field)
ASSAY_DEFS = {
    "ab_titer": {
        "table": "plasma_ab_titer",
        "var_fields": ["isotype", "antigen"],
        "value_field": "MFI_normalised",
    },
    "gene_expr": {
        "table": "pbmc_gene_expression",
        "var_fields": ["versioned_ensembl_gene_id"],
        "value_field": "tpm",
    },
    "cytokine_olink": {
        "table": "plasma_cytokine_concentration_by_olink",
        "var_fields": ["protein_id"],
        "value_field": "concentration",
    },
    "cytokine_legendplex": {
        "table": "plasma_cytokine_concentration_by_legendplex",
        "var_fields": ["protein_id"],
        "value_field": "concentration",
    },
    "cell_freq": {
        "table": "pbmc_cell_frequency",
        "var_fields": ["cell_type_name"],
        "value_field": "percent_live_cell",
    },
    "t_cell_polarization": {
        "table": "t_cell_polarization",
        "var_fields": ["protein_id", "stimulation"],
        "value_field": "analyte_counts",
    },
    "t_cell_activation": {
        "table": "t_cell_activation",
        "var_fields": ["stimulation"],
        "value_field": "analyte_percentages",
    },
}


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def fetch_table(table_name: str, params: Optional[Dict] = None,
                page_size: int = 5000) -> List[dict]:
    """
    Fetch all rows from a CMI-PB API table with pagination and caching.

    Parameters
    ----------
    table_name : API table name (e.g., "specimen")
    params : extra query parameters (e.g., {"planned_day_relative_to_boost": "eq.0"})
    page_size : rows per request

    Returns
    -------
    List of dicts (JSON rows).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Build cache key from table + params
    cache_suffix = ""
    if params:
        cache_suffix = "_" + "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    cache_path = os.path.join(CACHE_DIR, f"{table_name}{cache_suffix}.json")

    if os.path.exists(cache_path):
        print(f"  [cache] Loading {table_name} from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    print(f"  [fetch] Downloading {table_name} (page_size={page_size})...")
    all_rows = []
    offset = 0

    while True:
        query = dict(params) if params else {}
        query["limit"] = str(page_size)
        query["offset"] = str(offset)

        url = f"{API_BASE}/{table_name}?{urllib.parse.urlencode(query)}"
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if not data:
            break

        all_rows.extend(data)
        print(f"    fetched {len(all_rows)} rows so far...")

        if len(data) < page_size:
            break
        offset += page_size

    print(f"  [done] {table_name}: {len(all_rows)} total rows")
    with open(cache_path, "w") as f:
        json.dump(all_rows, f)

    return all_rows


# ---------------------------------------------------------------------------
# Specimen / subject helpers
# ---------------------------------------------------------------------------

def build_subject_specimen_map(
    days: List[int] = [0, 1, 3, 14],
) -> Tuple[Dict[int, Dict[int, int]], List[dict]]:
    """
    Build mapping: subject_id -> {day: specimen_id}.

    Only includes subjects that have a specimen at EVERY requested day.

    Returns
    -------
    mapping : {subject_id: {day: specimen_id}}
    specimens : raw specimen rows (for later lookups)
    """
    specimens = fetch_table("specimen")
    subjects = fetch_table("subject")

    # Build subject metadata lookup
    subject_meta = {s["subject_id"]: s for s in subjects}

    # Group by subject_id and planned_day
    subject_day_spec = defaultdict(dict)
    for row in specimens:
        sid = row["subject_id"]
        day = row["planned_day_relative_to_boost"]
        spec_id = row["specimen_id"]
        if day in days:
            subject_day_spec[sid][day] = spec_id

    # Keep only subjects with ALL days
    complete = {}
    for sid, day_map in subject_day_spec.items():
        if all(d in day_map for d in days):
            complete[sid] = day_map

    print(f"\nSpecimen map: {len(subject_day_spec)} subjects have at least one day")
    print(f"  {len(complete)} subjects have all {len(days)} days: {days}")

    return complete, specimens


# ---------------------------------------------------------------------------
# Per-assay data loading
# ---------------------------------------------------------------------------

def _make_var_key(row: dict, var_fields: List[str]) -> str:
    """Create a variable key from one or more fields."""
    return "|".join(str(row.get(f, "")) for f in var_fields)


def load_assay_for_specimens(
    assay_name: str,
    specimen_ids: set,
) -> Tuple[Dict[int, Dict[str, float]], List[str]]:
    """
    Load assay data and pivot to wide format.

    Returns
    -------
    data : {specimen_id: {variable_key: value}}
    variables : sorted list of unique variable keys
    """
    defn = ASSAY_DEFS[assay_name]
    table = defn["table"]
    var_fields = defn["var_fields"]
    value_field = defn["value_field"]

    rows = fetch_table(table)

    # Pivot
    data = defaultdict(dict)
    all_vars = set()

    for row in rows:
        spec_id = row["specimen_id"]
        if spec_id not in specimen_ids:
            continue

        var_key = _make_var_key(row, var_fields)
        val = row.get(value_field)

        # Skip null/missing values
        if val is None or (isinstance(val, str) and val.lower() in ("nan", "null", "")):
            continue

        try:
            val = float(val)
        except (ValueError, TypeError):
            continue

        data[spec_id][var_key] = val
        all_vars.add(var_key)

    variables = sorted(all_vars)
    print(f"  {assay_name}: {len(data)} specimens, {len(variables)} variables")

    return dict(data), variables


# ---------------------------------------------------------------------------
# Tensor assembly
# ---------------------------------------------------------------------------

def build_cmipb_tensor(
    days: List[int] = [0, 1, 3, 14],
    assays: Optional[List[str]] = None,
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Fetch CMI-PB data and assemble a block tensor.

    Parameters
    ----------
    days : list of planned_day_relative_to_boost values
    assays : list of assay names to include (default: all 7)
    cache_path : path to save/load assembled tensor (.npz)

    Returns
    -------
    X : (m, p, n) tensor
    b : (k,) block start indices
    meta : dict with variable_names, subject_ids, assay_names, block_sizes,
           subject_meta, missingness_report
    """
    if assays is None:
        assays = list(ASSAY_DEFS.keys())

    if cache_path is None:
        cache_path = os.path.join(CACHE_DIR, "tensor.npz")

    # Check for cached tensor
    if os.path.exists(cache_path):
        print(f"Loading cached tensor from {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        return loaded["X"], loaded["b"], loaded["meta"].item()

    print("=" * 60)
    print("Building CMI-PB block tensor")
    print("=" * 60)

    # Step 1: Get subject-specimen map
    print("\n--- Step 1: Subject-specimen mapping ---")
    subj_spec_map, specimens_raw = build_subject_specimen_map(days)

    # Collect all specimen IDs we need
    all_spec_ids = set()
    for day_map in subj_spec_map.values():
        all_spec_ids.update(day_map.values())

    # Step 2: Load each assay
    print("\n--- Step 2: Loading assays ---")
    assay_data = {}     # assay_name -> {specimen_id: {var: val}}
    assay_vars = {}     # assay_name -> [var_names]

    for name in assays:
        data, variables = load_assay_for_specimens(name, all_spec_ids)
        assay_data[name] = data
        assay_vars[name] = variables

    # Step 3: Find complete cases
    print("\n--- Step 3: Finding complete cases ---")
    subjects = sorted(subj_spec_map.keys())

    # For each subject, check if they have data in ALL assays at ALL days
    missingness = {name: 0 for name in assays}
    complete_subjects = []

    for sid in subjects:
        complete = True
        for name in assays:
            for day in days:
                spec_id = subj_spec_map[sid][day]
                if spec_id not in assay_data[name]:
                    missingness[name] += 1
                    complete = False
                    break
            if not complete:
                break
        if complete:
            complete_subjects.append(sid)

    print(f"\n  Subjects with all days: {len(subjects)}")
    print(f"  Complete cases (all {len(assays)} assays): {len(complete_subjects)}")
    print(f"\n  Missingness by assay (subjects dropped):")
    for name in assays:
        print(f"    {name}: {missingness[name]} subjects missing")

    if len(complete_subjects) < 10:
        print(f"\n  WARNING: Only {len(complete_subjects)} complete cases!")
        print("  Consider dropping assays with low coverage.")

    # Step 4: Assemble tensor
    print(f"\n--- Step 4: Assembling tensor ---")
    m = len(complete_subjects)
    n = len(days)

    # Compute block sizes and total p
    block_sizes = []
    for name in assays:
        block_sizes.append(len(assay_vars[name]))
    p = sum(block_sizes)

    # Block start indices
    b = np.zeros(len(assays), dtype=int)
    cumsum = 0
    for i, bs in enumerate(block_sizes):
        b[i] = cumsum
        cumsum += bs

    # Fill tensor
    X = np.zeros((m, p, n), dtype=np.float64)

    for i_subj, sid in enumerate(complete_subjects):
        for i_day, day in enumerate(days):
            spec_id = subj_spec_map[sid][day]
            col_offset = 0

            for name in assays:
                variables = assay_vars[name]
                spec_data = assay_data[name].get(spec_id, {})

                for j, var in enumerate(variables):
                    val = spec_data.get(var, 0.0)
                    X[i_subj, col_offset + j, i_day] = val

                col_offset += len(variables)

    # Build variable names list
    all_variable_names = []
    for name in assays:
        for var in assay_vars[name]:
            all_variable_names.append(f"{name}:{var}")

    # Load subject metadata
    subject_rows = fetch_table("subject")
    subject_meta = {s["subject_id"]: s for s in subject_rows}

    # Compute group labels (wP/aP vaccine type)
    vac_map = {"wP": 0, "aP": 1}
    subject_labels = np.array([
        vac_map.get(subject_meta.get(sid, {}).get("infancy_vac"), -1)
        for sid in complete_subjects
    ])

    meta = {
        "variable_names": all_variable_names,
        "subject_ids": complete_subjects,
        "assay_names": assays,
        "block_sizes": block_sizes,
        "days": days,
        "subject_meta": {sid: subject_meta.get(sid, {}) for sid in complete_subjects},
        "missingness_report": missingness,
        "subject_labels": subject_labels,
        "group_names": {0: "wP", 1: "aP"},
    }

    print(f"\n  Tensor shape: ({m}, {p}, {n})")
    print(f"  Blocks ({len(assays)}):")
    for i, name in enumerate(assays):
        start = b[i]
        end = b[i + 1] if i + 1 < len(b) else p
        print(f"    {name}: b[{i}]={start}, {end-start} variables")

    # Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, X=X, b=b, meta=meta)
    print(f"\n  Cached to {cache_path}")

    return X, b, meta


# ---------------------------------------------------------------------------
# Missingness report
# ---------------------------------------------------------------------------

def report_missingness(days: List[int] = [0, 1, 3, 14]):
    """Print detailed missingness statistics without building the tensor."""
    print("=" * 60)
    print("CMI-PB Missingness Report")
    print("=" * 60)

    subj_spec_map, _ = build_subject_specimen_map(days)
    all_spec_ids = set()
    for day_map in subj_spec_map.values():
        all_spec_ids.update(day_map.values())

    subjects = sorted(subj_spec_map.keys())
    assay_names = list(ASSAY_DEFS.keys())

    # Load all assays and check coverage
    coverage = {}  # assay -> set of subject_ids with data
    for name in assay_names:
        data, _ = load_assay_for_specimens(name, all_spec_ids)
        covered = set()
        for sid in subjects:
            has_all_days = True
            for day in days:
                spec_id = subj_spec_map[sid][day]
                if spec_id not in data:
                    has_all_days = False
                    break
            if has_all_days:
                covered.add(sid)
        coverage[name] = covered
        print(f"  {name}: {len(covered)}/{len(subjects)} subjects complete")

    # Intersection sizes
    print(f"\n  All {len(assay_names)} assays: "
          f"{len(set.intersection(*coverage.values()))} subjects")

    # Try dropping assays one at a time
    for drop in assay_names:
        remaining = [n for n in assay_names if n != drop]
        inter = set.intersection(*(coverage[n] for n in remaining))
        print(f"  Without {drop}: {len(inter)} subjects")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CMI-PB data pipeline")
    parser.add_argument("--days", nargs="+", type=int, default=[0, 1, 3, 14])
    parser.add_argument("--report-only", action="store_true",
                        help="Only report missingness, don't build tensor")
    args = parser.parse_args()

    if args.report_only:
        report_missingness(args.days)
    else:
        X, b, meta = build_cmipb_tensor(days=args.days)
        print(f"\nDone! Tensor shape: {X.shape}")
        print(f"Block vector: {b}")
        print(f"Subjects: {len(meta['subject_ids'])}")
