"""
Suez2018 Dataset Pipeline — load the post-antibiotic gut microbiome dataset.

Source: Suez et al. (2018), Cell 174(6):1406-1423.
Data hosted at: https://github.com/UriaMorP/tcam_analysis_notebooks

Returns (X, b, metadata) in the standard pipeline format.
"""

import os
import urllib.request
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(_REPO_ROOT, "data", "tcam")
SUEZ_URL = ("https://raw.githubusercontent.com/UriaMorP/tcam_analysis_notebooks"
            "/main/Suez2018/Suez2018.txt")


def _download_suez2018() -> str:
    """Download Suez2018.txt if not cached. Returns local path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, "Suez2018.txt")
    if os.path.exists(local_path):
        return local_path
    print(f"  [download] {SUEZ_URL}")
    urllib.request.urlretrieve(SUEZ_URL, local_path)
    print(f"  [saved] {local_path}")
    return local_path


# ---------------------------------------------------------------------------
# Genus -> Phylum lookup for block construction
# ---------------------------------------------------------------------------

GENUS_TO_PHYLUM = {
    # Firmicutes
    "Vagococcus": "Firmicutes", "Megasphaera": "Firmicutes",
    "Leuconostoc": "Firmicutes", "Streptococcus": "Firmicutes",
    "Tyzzerella": "Firmicutes", "Peptostreptococcus": "Firmicutes",
    "Lactobacillus": "Firmicutes", "Clostridium": "Firmicutes",
    "Ruminococcus": "Firmicutes", "Faecalibacterium": "Firmicutes",
    "Roseburia": "Firmicutes", "Eubacterium": "Firmicutes",
    "Blautia": "Firmicutes", "Coprococcus": "Firmicutes",
    "Dorea": "Firmicutes", "Lachnospira": "Firmicutes",
    "Anaerostipes": "Firmicutes", "Dialister": "Firmicutes",
    "Veillonella": "Firmicutes", "Enterococcus": "Firmicutes",
    "Staphylococcus": "Firmicutes", "Bacillus": "Firmicutes",
    "Listeria": "Firmicutes", "Erysipelatoclostridium": "Firmicutes",
    "Clostridioides": "Firmicutes", "Lacticaseibacillus": "Firmicutes",
    "Limosilactobacillus": "Firmicutes", "Lactiplantibacillus": "Firmicutes",
    "Agathobacter": "Firmicutes", "Subdoligranulum": "Firmicutes",
    "Oscillibacter": "Firmicutes", "Flavonifractor": "Firmicutes",
    "Butyricicoccus": "Firmicutes", "Anaerobutyricum": "Firmicutes",
    "Hungatella": "Firmicutes", "Intestinibacter": "Firmicutes",
    "Eisenbergiella": "Firmicutes", "Sellimonas": "Firmicutes",
    "Fusicatenibacter": "Firmicutes", "Lachnoclostridium": "Firmicutes",
    "Mediterraneibacter": "Firmicutes", "Monoglobus": "Firmicutes",
    "Asaccharobacter": "Firmicutes",
    # Bacteroidetes
    "Bacteroides": "Bacteroidetes", "Prevotella": "Bacteroidetes",
    "Parabacteroides": "Bacteroidetes", "Alistipes": "Bacteroidetes",
    "Barnesiella": "Bacteroidetes", "Odoribacter": "Bacteroidetes",
    "Porphyromonas": "Bacteroidetes", "Tannerella": "Bacteroidetes",
    "Dysgonomonas": "Bacteroidetes", "Paraprevotella": "Bacteroidetes",
    "Phocaeicola": "Bacteroidetes",
    # Proteobacteria
    "Escherichia": "Proteobacteria", "Klebsiella": "Proteobacteria",
    "Salmonella": "Proteobacteria", "Shigella": "Proteobacteria",
    "Enterobacter": "Proteobacteria", "Citrobacter": "Proteobacteria",
    "Proteus": "Proteobacteria", "Haemophilus": "Proteobacteria",
    "Helicobacter": "Proteobacteria", "Campylobacter": "Proteobacteria",
    "Desulfovibrio": "Proteobacteria", "Bilophila": "Proteobacteria",
    "Sutterella": "Proteobacteria", "Parasutterella": "Proteobacteria",
    # Actinobacteria
    "Bifidobacterium": "Actinobacteria", "Collinsella": "Actinobacteria",
    "Eggerthella": "Actinobacteria", "Slackia": "Actinobacteria",
    "Adlercreutzia": "Actinobacteria", "Gordonibacter": "Actinobacteria",
    "Actinomyces": "Actinobacteria",
    # Verrucomicrobia
    "Akkermansia": "Verrucomicrobia",
    # Fusobacteria
    "Fusobacterium": "Fusobacteria",
}


def _get_genus(taxon_name: str) -> str:
    """Extract genus from 's__Genus_species' format."""
    name = taxon_name
    if name.startswith("s__"):
        name = name[3:]
    parts = name.split("_")
    return parts[0] if parts else name


def _assign_phylum_blocks(taxa_names):
    """
    Partition taxa into blocks by phylum.

    Returns
    -------
    b : block start indices
    block_names : list of phylum names
    taxa_order : permutation indices to reorder columns by phylum
    """
    phylum_list = [GENUS_TO_PHYLUM.get(_get_genus(n), "Other") for n in taxa_names]

    phylum_to_indices = {}
    for i, ph in enumerate(phylum_list):
        phylum_to_indices.setdefault(ph, []).append(i)

    sorted_phyla = sorted(
        [p for p in phylum_to_indices if p != "Other"],
        key=lambda p: -len(phylum_to_indices[p])
    )
    if "Other" in phylum_to_indices:
        sorted_phyla.append("Other")

    taxa_order = []
    block_starts = []
    block_names = []
    offset = 0

    for ph in sorted_phyla:
        indices = phylum_to_indices[ph]
        if not indices:
            continue
        block_starts.append(offset)
        block_names.append(f"{ph} ({len(indices)})")
        taxa_order.extend(indices)
        offset += len(indices)

    return np.array(block_starts, dtype=int), block_names, np.array(taxa_order, dtype=int)


def load_suez2018(log_fold_baseline: bool = True):
    """
    Load Suez2018 dataset as a block tensor.

    Parameters
    ----------
    log_fold_baseline : bool
        If True, apply log2(x + 1e-6) and subtract baseline (timepoint 0).

    Returns
    -------
    X : (m, p, n) tensor -- subjects x taxa x timepoints
    b : block start indices (phylum-level blocks)
    metadata : dict with keys:
        - name, taxa_names, block_names, subject_labels, group_names,
          is_longitudinal, is_multi_omic, mode1_map, mode3_map
    """
    path = _download_suez2018()

    data_raw = pd.read_csv(path, sep="\t", index_col=list(range(6)))
    data_raw.rename(index={"Sp": "CTR", "Prob": "PBX"}, level="rGroup",
                    inplace=True)

    taxa_names = list(data_raw.columns)

    # Save group info before dropping index levels
    participant_groups = {}
    for idx in data_raw.index:
        participant_groups[idx[0]] = idx[4]

    data_2level = data_raw.copy()
    data_2level.index = pd.MultiIndex.from_arrays(
        [data_raw.index.get_level_values("Participant"),
         data_raw.index.get_level_values("rDay")],
        names=["Participant", "rDay"]
    )

    try:
        from mprod import table2tensor
    except ImportError as e:
        raise ImportError(
            "load_suez2018 requires the optional 'mprod' package. "
            "Install with: pip install mprod"
        ) from e
    tensor, mode1_map, mode3_map = table2tensor(data_2level, missing_flag=True)
    X = np.array(tensor, dtype=np.float64)

    if hasattr(tensor, 'mask'):
        mask = np.array(tensor.mask)
        if mask.any():
            X[mask] = 0.0

    # Log-fold baseline normalization
    if log_fold_baseline:
        X = np.log2(X + 1e-6)
        X = X - X[:, :, 0:1]

    # Build phylum blocks and reorder taxa
    b, block_names, taxa_order = _assign_phylum_blocks(taxa_names)
    X = X[:, taxa_order, :]
    taxa_names_ordered = [taxa_names[i] for i in taxa_order]

    # Group labels
    group_map = {"CTR": 0, "FMT": 1, "PBX": 2}
    group_names = {0: "CTR", 1: "FMT", 2: "PBX"}

    m = X.shape[0]
    subject_labels = np.zeros(m, dtype=int)
    for participant_id, tidx in mode1_map.items():
        grp = participant_groups.get(participant_id)
        if grp is not None:
            subject_labels[tidx] = group_map.get(grp, -1)

    metadata = {
        "name": "suez2018",
        "description": "Post-antibiotic gut microbiome reconstitution (16S rRNA)",
        "taxa_names": taxa_names_ordered,
        "block_names": block_names,
        "subject_labels": subject_labels,
        "group_names": group_names,
        "is_longitudinal": True,
        "is_multi_omic": False,
        "mode1_map": mode1_map,
        "mode3_map": mode3_map,
    }

    return X, b, metadata
