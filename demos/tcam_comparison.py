"""
Suez2018 Comparison — run TBI-I, TBI-II, and Matrix MCIA on the Suez2018
dataset with bacterial taxa partitioned into phylum-level blocks.

Usage
-----
    cd TBI
    conda activate claude
    python tcam_comparison.py
"""

import os
import time
import urllib.request
import numpy as np
import pandas as pd

from mprod import table2tensor

from TBI import TBI_I, TBI_II, matrix_MCIA
from TBI.normalization import default_normalize, no_normalize
from TBI.analysis_utils import (
    dct_matrix, scree_plot, score_scatter_grid,
    block_contribution_bar, block_variance_contributions,
    efficiency_plot,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tcam")
SUEZ_URL = ("https://raw.githubusercontent.com/UriaMorP/tcam_analysis_notebooks"
            "/main/Suez2018/Suez2018.txt")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def download_suez2018() -> str:
    """Download Suez2018.txt if not cached. Returns local path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, "Suez2018.txt")
    if os.path.exists(local_path):
        print(f"  [cache] {local_path}")
        return local_path

    print(f"  [download] {SUEZ_URL}")
    urllib.request.urlretrieve(SUEZ_URL, local_path)
    print(f"  [saved] {local_path}")
    return local_path


def load_suez2018():
    """
    Load Suez2018 dataset and return tensor + metadata.

    Returns
    -------
    X : (m, p, n) tensor — subjects x taxa x timepoints
    taxa_names : list of p taxa names
    subject_labels : (m,) array of group labels (0=CTR, 1=FMT, 2=PBX)
    group_names : {0: "CTR", 1: "FMT", 2: "PBX"}
    mode1_map : {tensor_idx: subject_name}
    mode3_map : {tensor_idx: day}
    """
    path = download_suez2018()

    # Load with multi-index (6 index columns)
    data_raw = pd.read_csv(path, sep="\t", index_col=list(range(6)))
    data_raw.rename(index={"Sp": "CTR", "Prob": "PBX"}, level="rGroup",
                    inplace=True)

    taxa_names = list(data_raw.columns)

    # table2tensor expects exactly 2 index levels: (subject, timepoint)
    # Keep Participant (level 0) and rDay (level 5), drop others
    # First, save the group info before dropping levels
    participant_groups = {}
    for idx in data_raw.index:
        participant_groups[idx[0]] = idx[4]  # Participant -> rGroup

    data_2level = data_raw.copy()
    data_2level.index = pd.MultiIndex.from_arrays(
        [data_raw.index.get_level_values("Participant"),
         data_raw.index.get_level_values("rDay")],
        names=["Participant", "rDay"]
    )

    tensor, mode1_map, mode3_map = table2tensor(data_2level, missing_flag=True)
    X = np.array(tensor, dtype=np.float64)

    # Handle masked values (impute with 0 for now)
    if hasattr(tensor, 'mask'):
        mask = np.array(tensor.mask)
        if mask.any():
            print(f"  Warning: {mask.sum()} masked values, imputing with 0")
            X[mask] = 0.0

    print(f"  Tensor shape: {X.shape} (subjects x taxa x timepoints)")
    print(f"  Mode-1 (subjects): {mode1_map}")
    print(f"  Mode-3 (timepoints): {mode3_map}")

    # Extract group labels
    group_map = {"CTR": 0, "FMT": 1, "PBX": 2}
    group_names = {0: "CTR", 1: "FMT", 2: "PBX"}

    # participant_groups was built before we dropped index levels
    subject_groups = {}
    for participant, rgroup in participant_groups.items():
        subject_groups[participant] = group_map.get(rgroup, -1)

    # Map to tensor indices
    # mode1_map is {participant_id: tensor_index}
    m = X.shape[0]
    subject_labels = np.zeros(m, dtype=int)
    for participant_id, tidx in mode1_map.items():
        if participant_id in subject_groups:
            subject_labels[tidx] = subject_groups[participant_id]

    return X, taxa_names, subject_labels, group_names, mode1_map, mode3_map


# ---------------------------------------------------------------------------
# Phylum block construction
# ---------------------------------------------------------------------------

def assign_phylum_blocks(taxa_names):
    """
    Partition taxa into blocks by phylum using genus→phylum lookup.

    Suez2018 taxa names are species-level: "s__Genus_species".
    We map the genus to its phylum using a lookup of common gut bacteria.

    Returns
    -------
    b : block start indices
    block_names : list of phylum names
    taxa_order : permutation indices to reorder columns by phylum
    """
    # Genus → Phylum lookup for common gut bacteria
    # Covers the major genera found in human gut microbiome studies
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

    def get_genus(taxon_name):
        """Extract genus from 's__Genus_species' format."""
        name = taxon_name
        if name.startswith("s__"):
            name = name[3:]
        parts = name.split("_")
        return parts[0] if parts else name

    phylum_list = []
    for name in taxa_names:
        genus = get_genus(name)
        phylum = GENUS_TO_PHYLUM.get(genus, "Other")
        phylum_list.append(phylum)

    # Group taxa by phylum
    phylum_to_indices = {}
    for i, ph in enumerate(phylum_list):
        if ph not in phylum_to_indices:
            phylum_to_indices[ph] = []
        phylum_to_indices[ph].append(i)

    # Sort phyla by size (largest first), keep "Other" at end
    sorted_phyla = sorted(
        [p for p in phylum_to_indices if p != "Other"],
        key=lambda p: -len(phylum_to_indices[p])
    )
    if "Other" in phylum_to_indices:
        sorted_phyla.append("Other")

    # Build permutation and block indices
    taxa_order = []
    block_starts = []
    block_names = []

    offset = 0
    for ph in sorted_phyla:
        indices = phylum_to_indices[ph]
        if len(indices) == 0:
            continue
        block_starts.append(offset)
        block_names.append(f"{ph} ({len(indices)})")
        taxa_order.extend(indices)
        offset += len(indices)

    b = np.array(block_starts, dtype=int)
    taxa_order = np.array(taxa_order, dtype=int)

    print(f"\n  Phylum blocks ({len(block_names)}):")
    for i, name in enumerate(block_names):
        start = b[i]
        end = b[i + 1] if i + 1 < len(b) else len(taxa_names)
        print(f"    {name}: [{start}:{end}]")

    return b, block_names, taxa_order


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_suez2018_comparison(energy=0.99, max_components=20):
    """Run TBI-I, TBI-II, and Matrix MCIA on Suez2018 and compare."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("Suez2018: TBI vs Matrix MCIA Comparison")
    print("=" * 60)

    # Load data
    print("\n--- Loading Suez2018 ---")
    X_raw, taxa_names, subject_labels, group_names, m1map, m3map = load_suez2018()
    m, p, n = X_raw.shape

    # Log-fold baseline normalization (as in TCAM paper)
    # Add pseudocount, take log, subtract baseline (first timepoint)
    X = X_raw.copy()
    X = np.log2(X + 1e-6)
    baseline = X[:, :, 0:1]  # (m, p, 1)
    X = X - baseline

    print(f"\n  After log-fold baseline: shape {X.shape}")

    # Build phylum blocks
    print("\n--- Building phylum blocks ---")
    b, block_names, taxa_order = assign_phylum_blocks(taxa_names)

    # Reorder taxa by phylum
    X_blocked = X[:, taxa_order, :]
    taxa_names_ordered = [taxa_names[i] for i in taxa_order]

    # DCT matrix
    M = dct_matrix(n)

    # --- Run TBI-I with blocks ---
    print("\n--- Running TBI-I (phylum blocks) ---")
    start = time.perf_counter()
    result_I = TBI_I(X_blocked, b, M, energy=energy, max_iter=max_components)
    tbi_I_time = time.perf_counter() - start
    print(f"  TBI-I: {result_I.n_iter} iterations, elapsed={tbi_I_time:.3f}s")
    print(f"    Total variance: {result_I.total_variance:.2f}")
    print(f"    Variance explained: {result_I.variance_explained[:5]}")

    # --- Run TBI-II with blocks ---
    print("\n--- Running TBI-II (phylum blocks, greedy) ---")
    start = time.perf_counter()
    result_II = TBI_II(X_blocked, b, M, energy=energy, max_iter=max_components * n)
    tbi_II_time = time.perf_counter() - start
    print(f"  TBI-II: {result_II.n_iter} iterations, elapsed={tbi_II_time:.3f}s")
    print(f"    Total variance: {result_II.total_variance:.2f}")
    print(f"    Variance explained: {result_II.variance_explained[:5]}")
    print(f"    Sheet selections: {result_II.sheet_indices}")

    # --- Matrix MCIA baseline ---
    print("\n--- Running Matrix MCIA (phylum blocks) ---")
    start = time.perf_counter()
    result_mcia = matrix_MCIA(X_blocked, b, energy=energy, max_iter=max_components)
    mcia_time = time.perf_counter() - start
    print(f"  Matrix MCIA: {result_mcia.n_iter} iterations, elapsed={mcia_time:.3f}s")
    print(f"    Total variance: {result_mcia.total_variance:.2f}")
    print(f"    Variance explained: {result_mcia.variance_explained[:5]}")

    # --- Comparison plots ---
    print("\n--- Generating comparison plots ---")

    # 1. Scree plot comparison
    n_compare = min(max_components, result_I.n_iter)
    scree_plot(
        {
            "TBI-I": result_I.variance_explained[:n_compare],
            "TBI-II": result_II.variance_explained[:n_compare],
            "Matrix MCIA": result_mcia.variance_explained[:n_compare],
        },
        total_variances={
            "TBI-I": result_I.total_variance,
            "TBI-II": result_II.total_variance,
            "Matrix MCIA": result_mcia.total_variance,
        },
        title="Suez2018: TBI vs Matrix MCIA",
        savepath=os.path.join(FIGURES_DIR, "suez2018_scree.png"),
    )

    # 2. Score scatter grid — all methods side by side
    scores_dict = {}
    if result_I.n_iter >= 2:
        scores_dict["TBI-I"] = result_I.global_scores[:, :2, 0]
    if result_II.n_iter >= 2:
        scores_dict["TBI-II"] = result_II.global_scores[:, :2]
    if result_mcia.n_iter >= 2:
        scores_dict["Matrix MCIA"] = result_mcia.scores[:, :2]

    if scores_dict:
        score_scatter_grid(
            scores_dict, subject_labels, group_names,
            title="Suez2018: Score Comparison",
            savepath=os.path.join(FIGURES_DIR, "suez2018_scores_grid.png"),
        )

    # 3. Block contribution — before vs after TBI normalization
    from TBI.star_M import mode3
    X_hat = mode3(X_blocked, M)
    fracs_before = block_variance_contributions(X_hat, b)

    X_hat_norm = default_normalize(X_hat, b)
    fracs_after = block_variance_contributions(X_hat_norm, b)

    block_contribution_bar(
        fracs_before, block_names,
        title="Suez2018: Block Variance BEFORE TBI Normalization",
        savepath=os.path.join(FIGURES_DIR, "suez2018_blocks_before.png"),
    )
    block_contribution_bar(
        fracs_after, block_names,
        title="Suez2018: Block Variance AFTER Normalization",
        savepath=os.path.join(FIGURES_DIR, "suez2018_blocks_after.png"),
    )

    # 5. Efficiency plot
    n_eff = min(max_components, result_I.n_iter)
    efficiency_plot(
        {
            "TBI-I": result_I.variance_explained[:n_eff],
            "TBI-II": result_II.variance_explained,
            "Matrix MCIA": result_mcia.variance_explained[:n_eff],
        },
        total_variances={
            "TBI-I": result_I.total_variance,
            "TBI-II": result_II.total_variance,
            "Matrix MCIA": result_mcia.total_variance,
        },
        storage_per_component={
            "TBI-I": (m + p) * n, "TBI-II": m + p,
            "Matrix MCIA": m + p * n,
        },
        title="Suez2018: Variance Explained vs Storage (Scores + Loadings)",
        savepath=os.path.join(FIGURES_DIR, "suez2018_efficiency.png"),
    )

    # 6. Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<12} {'Components':>10} {'Time (s)':>10} {'Cum Var %':>10}")
    print("-" * 42)

    tbi_I_cum = result_I.variance_explained.sum() / result_I.total_variance * 100
    tbi_II_cum = result_II.variance_explained.sum() / result_II.total_variance * 100
    mcia_cum = result_mcia.variance_explained.sum() / result_mcia.total_variance * 100

    print(f"{'TBI-I':<12} {result_I.n_iter:>10d} {tbi_I_time:>10.3f} {tbi_I_cum:>9.1f}%")
    print(f"{'TBI-II':<12} {result_II.n_iter:>10d} {tbi_II_time:>10.3f} {tbi_II_cum:>9.1f}%")
    print(f"{'Matrix MCIA':<12} {result_mcia.n_iter:>10d} {mcia_time:>10.3f} {mcia_cum:>9.1f}%")


if __name__ == "__main__":
    run_suez2018_comparison()
