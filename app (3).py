"""
Distributed Peer Review (DPR) Allocator
Supports random allocation, TF-IDF semantic matching, and sentence-embedding semantic matching.
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import io
import re
import zipfile
from collections import Counter
from math import log

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DPR Allocator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem; color: #1a1a2e;
    line-height: 1.15; margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1.05rem; color: #555;
    font-weight: 300; margin-bottom: 2rem;
}
.step-card {
    background: #f7f7fb; border-left: 4px solid #4f46e5;
    border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
.step-number {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #4f46e5;
}
.step-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem; color: #1a1a2e; margin: 0;
}
section[data-testid="stSidebar"] { background: #1a1a2e; }
section[data-testid="stSidebar"] * { color: #e2e2f0 !important; }
.stDownloadButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white !important; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.55rem 1.4rem; transition: opacity 0.2s;
}
.stDownloadButton > button:hover { opacity: 0.88; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; font-size: 1rem;
    padding: 0.65rem 2rem; width: 100%;
}
hr { border-color: #e5e5ef; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  TF-IDF SIMILARITY ENGINE  (no external ML libraries)
# ═══════════════════════════════════════════════════════

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "this","that","these","those","it","its","we","our","they","their",
    "which","who","what","how","when","where","as","than","also","not",
    "no","so","if","can","each","both","all","any","more","such","into",
    "through","during","after","before","between","within","while","about",
    "against","further","then","once","here","there","up","down","out","over",
}

def tokenise(text: str) -> list:
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def build_tfidf_matrix(corpus: list) -> np.ndarray:
    n = len(corpus)
    tokenised = [tokenise(doc) for doc in corpus]
    vocab = sorted({tok for doc in tokenised for tok in doc})
    vocab_index = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    tf = np.zeros((n, V), dtype=np.float32)
    for d, tokens in enumerate(tokenised):
        counts = Counter(tokens)
        for w, c in counts.items():
            if w in vocab_index:
                tf[d, vocab_index[w]] = 1 + log(c)

    df = (tf > 0).sum(axis=0)
    idf = np.log((n + 1) / (df + 1)) + 1
    tfidf = tf * idf

    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tfidf /= norms
    return tfidf


def cosine_similarity_matrix(tfidf: np.ndarray) -> np.ndarray:
    sim = tfidf @ tfidf.T
    np.fill_diagonal(sim, 0.0)
    return sim

# ═══════════════════════════════════════════════════════
#  SENTENCE-EMBEDDINGS SIMILARITY ENGINE
# ═══════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading embedding model (first run only — ~90 MB download)…")
def load_embedding_model():
    """Load and cache the sentence-transformer model. Downloaded once, reused every run."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_embeddings_matrix(corpus: list) -> np.ndarray:
    """
    Encode documents with a small sentence-transformer model and return
    an L2-normalised embedding matrix (n_docs x 384).
    Cosine similarity is then just embeddings @ embeddings.T
    """
    model = load_embedding_model()
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


# ═══════════════════════════════════════════════════════
#  CORE ALLOCATION LOGIC
# ═══════════════════════════════════════════════════════

def parse_conflicts(conflict_str) -> list:
    if pd.isna(conflict_str) or str(conflict_str).strip() == "":
        return []
    return [c.strip() for c in str(conflict_str).split(",") if c.strip()]


def generate_constraint_list(data1: pd.DataFrame, data2: pd.DataFrame, master: pd.DataFrame) -> dict:
    name_to_id = dict(zip(master["ApplicantName"], master["ID"]))
    constraints = {}
    for _, row in data1.iterrows():
        forbidden_ids = set()
        for cname in parse_conflicts(row.get("Conflicts", "")):
            if cname in name_to_id:
                cid = name_to_id[cname]
                if cid in data2["ID"].values:
                    forbidden_ids.add(cid)
        constraints[row["ID"]] = forbidden_ids
    return constraints


# ── Random allocation ──────────────────────────────────

def dpr_allocation_random(group_to_allocate, group_from, constraint_list, reviews_per_person=10):
    from_ids = list(group_from["ID"])
    count = {fid: 0 for fid in from_ids}
    assignments = {}
    allocate_order = list(group_to_allocate["ID"])
    random.shuffle(allocate_order)

    for reviewer_id in allocate_order:
        forbidden = constraint_list.get(reviewer_id, set())
        already = set(assignments.get(reviewer_id, []))
        available = [fid for fid in from_ids
                     if fid not in forbidden and fid not in already
                     and count[fid] < reviews_per_person]
        random.shuffle(available)
        chosen = available[:reviews_per_person]
        assignments[reviewer_id] = list(chosen)
        for c in chosen:
            count[c] += 1

    return assignments, count


# ── Semantic (TF-IDF) allocation ───────────────────────

def dpr_allocation_semantic(group_to_allocate, group_from, constraint_list,
                             sim_matrix, id_to_idx, reviews_per_person=10):
    """
    Greedy similarity-optimised allocation.
    Processes most-constrained reviewers first, assigns highest-similarity
    available reviewees subject to capacity and conflict constraints.
    """
    from_ids = list(group_from["ID"])
    count = {fid: 0 for fid in from_ids}
    assignments = {}

    # Most-constrained first
    def n_available(rid):
        forbidden = constraint_list.get(rid, set())
        return sum(1 for fid in from_ids
                   if fid not in forbidden and count[fid] < reviews_per_person)

    allocate_order = sorted(group_to_allocate["ID"].tolist(), key=n_available)

    for reviewer_id in allocate_order:
        forbidden = constraint_list.get(reviewer_id, set())
        already = set(assignments.get(reviewer_id, []))
        r_idx = id_to_idx[reviewer_id]

        candidates = []
        for fid in from_ids:
            if fid in forbidden or fid in already or count[fid] >= reviews_per_person:
                continue
            score = float(sim_matrix[r_idx, id_to_idx[fid]])
            candidates.append((fid, score))

        candidates.sort(key=lambda x: -x[1])
        chosen = [fid for fid, _ in candidates[:reviews_per_person]]

        assignments[reviewer_id] = chosen
        for c in chosen:
            count[c] += 1

    return assignments, count


# ── Redistribution (shared by both modes) ─────────────

def redistribute_incomplete(assignments, group_to_allocate, group_from,
                             constraint_list, count, reviews_per_person=10,
                             sim_matrix=None, id_to_idx=None):
    from_ids = set(group_from["ID"])
    max_iter = 500

    for _ in range(max_iter):
        short = [rid for rid, lst in assignments.items()
                 if len(lst) < reviews_per_person]
        if not short:
            break

        for short_reviewer in short:
            needed = reviews_per_person - len(assignments[short_reviewer])
            forbidden = constraint_list.get(short_reviewer, set())
            already = set(assignments[short_reviewer])

            available = [fid for fid in from_ids
                         if fid not in forbidden and fid not in already
                         and count.get(fid, 0) < reviews_per_person]

            if sim_matrix is not None and id_to_idx is not None:
                r_idx = id_to_idx[short_reviewer]
                available.sort(key=lambda fid: -float(sim_matrix[r_idx, id_to_idx[fid]]))
            else:
                random.shuffle(available)

            for a in available[:needed]:
                assignments[short_reviewer].append(a)
                count[a] = count.get(a, 0) + 1

            still_needed = reviews_per_person - len(assignments[short_reviewer])
            if still_needed <= 0:
                continue

            for donor_id, donor_list in assignments.items():
                if len(donor_list) < reviews_per_person:
                    continue
                for candidate in list(donor_list):
                    if candidate in forbidden or candidate in set(assignments[short_reviewer]):
                        continue
                    donor_forbidden = constraint_list.get(donor_id, set())
                    replacement_pool = [fid for fid in from_ids
                                        if fid not in donor_forbidden
                                        and fid not in set(donor_list)
                                        and fid != candidate
                                        and count.get(fid, 0) < reviews_per_person]
                    if not replacement_pool:
                        continue
                    replacement = random.choice(replacement_pool)
                    donor_list.remove(candidate)
                    donor_list.append(replacement)
                    count[replacement] = count.get(replacement, 0) + 1
                    assignments[short_reviewer].append(candidate)
                    still_needed -= 1
                    if still_needed <= 0:
                        break
                if still_needed <= 0:
                    break

    return assignments, count


def find_reciprocals(g1_assignments, group2):
    reciprocals = {row["ID"]: set() for _, row in group2.iterrows()}
    for reviewer_id, reviewee_list in g1_assignments.items():
        for reviewee_id in reviewee_list:
            if reviewee_id in reciprocals:
                reciprocals[reviewee_id].add(reviewer_id)
    return reciprocals


def build_final_df(assignments, master, reviews_per_person,
                   sim_matrix=None, id_to_idx=None):
    id_to_name   = dict(zip(master["ID"], master["ApplicantName"]))
    id_to_email  = dict(zip(master["ID"], master.get("Email",  pd.Series(dtype=str))))
    id_to_appnum = dict(zip(master["ID"], master.get("ApplicationNumber", master["ID"])))

    rows = []
    for reviewer_id, reviewees in assignments.items():
        row = {
            "Reviewer_ID":                reviewer_id,
            "Reviewer_Name":              id_to_name.get(reviewer_id, str(reviewer_id)),
            "Reviewer_Email":             id_to_email.get(reviewer_id, ""),
            "Reviewer_ApplicationNumber": id_to_appnum.get(reviewer_id, reviewer_id),
        }
        for i, rev_id in enumerate(reviewees, 1):
            row[f"Reviewee_{i}_Name"]   = id_to_name.get(rev_id, str(rev_id))
            row[f"Reviewee_{i}_Email"]  = id_to_email.get(rev_id, "")
            row[f"Reviewee_{i}_AppNum"] = id_to_appnum.get(rev_id, rev_id)
            if sim_matrix is not None and id_to_idx is not None:
                r_idx = id_to_idx[reviewer_id]
                f_idx = id_to_idx[rev_id]
                row[f"Reviewee_{i}_Similarity"] = round(float(sim_matrix[r_idx, f_idx]), 4)
        for i in range(len(reviewees) + 1, reviews_per_person + 1):
            row[f"Reviewee_{i}_Name"]   = ""
            row[f"Reviewee_{i}_Email"]  = ""
            row[f"Reviewee_{i}_AppNum"] = ""
            if sim_matrix is not None:
                row[f"Reviewee_{i}_Similarity"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def compute_allocation_quality(assignments, sim_matrix, id_to_idx):
    scores = []
    reviewer_means = {}
    for rid, reviewees in assignments.items():
        if not reviewees:
            continue
        r_idx = id_to_idx[rid]
        pair_scores = [float(sim_matrix[r_idx, id_to_idx[fid]]) for fid in reviewees]
        reviewer_means[rid] = float(np.mean(pair_scores))
        scores.extend(pair_scores)
    overall_mean = float(np.mean(scores)) if scores else 0.0
    overall_min  = float(np.min(scores))  if scores else 0.0
    return reviewer_means, overall_mean, overall_min


# ── Main pipeline ──────────────────────────────────────

def run_allocation(master: pd.DataFrame, reviews_per_person: int, seed: int,
                   mode: str = "random", floor_percentile: float = 20.0):
    random.seed(seed)
    np.random.seed(seed)

    sim_matrix = None
    id_to_idx  = None

    if mode in ("semantic", "embeddings"):
        if "ApplicationText" not in master.columns or master["ApplicationText"].fillna("").str.strip().eq("").all():
            raise ValueError("This mode requires a non-empty 'ApplicationText' column.")
        texts     = master["ApplicationText"].fillna("").tolist()
        ids       = master["ID"].tolist()
        id_to_idx = {aid: i for i, aid in enumerate(ids)}
        if mode == "embeddings":
            vecs       = build_embeddings_matrix(texts)
            sim_matrix = cosine_similarity_matrix(vecs)
        else:
            tfidf      = build_tfidf_matrix(texts)
            sim_matrix = cosine_similarity_matrix(tfidf)

    ids = list(master["ID"])
    random.shuffle(ids)
    half   = len(ids) // 2
    group1 = master[master["ID"].isin(set(ids[:half]))].reset_index(drop=True)
    group2 = master[master["ID"].isin(set(ids[half:]))].reset_index(drop=True)

    # ── Group 1 reviews Group 2 ──
    g1_constraints = generate_constraint_list(group1, group2, master)
    if mode in ("semantic", "embeddings"):
        g1_assignments, g2_count = dpr_allocation_semantic(
            group1, group2, g1_constraints, sim_matrix, id_to_idx, reviews_per_person)
    else:
        g1_assignments, g2_count = dpr_allocation_random(
            group1, group2, g1_constraints, reviews_per_person)

    g1_assignments, g2_count = redistribute_incomplete(
        g1_assignments, group1, group2, g1_constraints, g2_count,
        reviews_per_person, sim_matrix, id_to_idx)

    # ── Reciprocal constraints ──
    reciprocals = find_reciprocals(g1_assignments, group2)
    master2 = master.copy()
    id_to_name = dict(zip(master["ID"], master["ApplicantName"]))
    for g2_id, recip_ids in reciprocals.items():
        recip_names = [id_to_name.get(r, str(r)) for r in recip_ids]
        idx = master2[master2["ID"] == g2_id].index
        if len(idx):
            existing = master2.loc[idx[0], "Conflicts"]
            combined = list(set(parse_conflicts(existing) + recip_names))
            master2.loc[idx[0], "Conflicts"] = ",".join(combined)

    # ── Group 2 reviews Group 1 ──
    g2_constraints = generate_constraint_list(group2, group1, master2)
    if mode in ("semantic", "embeddings"):
        g2_assignments, g1_count = dpr_allocation_semantic(
            group2, group1, g2_constraints, sim_matrix, id_to_idx, reviews_per_person)
    else:
        g2_assignments, g1_count = dpr_allocation_random(
            group2, group1, g2_constraints, reviews_per_person)

    g2_assignments, g1_count = redistribute_incomplete(
        g2_assignments, group2, group1, g2_constraints, g1_count,
        reviews_per_person, sim_matrix, id_to_idx)

    # ── Output DataFrames ──
    df_g1 = build_final_df(g1_assignments, master, reviews_per_person, sim_matrix, id_to_idx)
    df_g2 = build_final_df(g2_assignments, master, reviews_per_person, sim_matrix, id_to_idx)
    df_combined = pd.concat([df_g1, df_g2], ignore_index=True)

    # ── Stats ──
    all_assignments = {**g1_assignments, **g2_assignments}
    review_counts = {}
    for reviewer_id, reviewees in all_assignments.items():
        for rid in reviewees:
            review_counts[rid] = review_counts.get(rid, 0) + 1

    quality = {}
    if mode in ("semantic", "embeddings") and sim_matrix is not None:
        reviewer_means, overall_mean, overall_min = compute_allocation_quality(
            all_assignments, sim_matrix, id_to_idx)
        quality = {
            "reviewer_means": reviewer_means,
            "overall_mean":   overall_mean,
            "overall_min":    overall_min,
            "sim_matrix":     sim_matrix,
            "id_to_idx":      id_to_idx,
            "ids":            master["ID"].tolist(),
            "names":          master["ApplicantName"].tolist(),
        }

    stats = {
        "total_applicants":          len(master),
        "group1_size":               len(group1),
        "group2_size":               len(group2),
        "full_reviews":              sum(1 for c in review_counts.values() if c >= reviews_per_person),
        "partial_reviews":           sum(1 for c in review_counts.values() if c < reviews_per_person),
        "review_count_distribution": review_counts,
        "quality":                   quality,
    }

    return df_combined, df_g1, df_g2, stats


# ═══════════════════════════════════════════════════════
#  TEMPLATE CSV  — 20 biology abstracts
# ═══════════════════════════════════════════════════════

BIOLOGY_ABSTRACTS = [
    ("CRISPR-Cas9 off-target mutagenesis in human embryonic stem cells",
     "We characterise genome-wide off-target cleavage events induced by CRISPR-Cas9 in human embryonic stem cells using unbiased GUIDE-seq. Whole-genome sequencing reveals that high-fidelity Cas9 variants reduce indel frequencies at predicted off-target loci by two orders of magnitude without compromising on-target editing efficiency. Transcriptomic analysis demonstrates that pluripotency gene networks remain intact following editing."),
    ("Structural basis of AMPA receptor gating and pharmacological modulation",
     "Cryo-electron microscopy structures of GluA2 homotetramers bound to positive allosteric modulators reveal a transmembrane domain rearrangement that prolongs channel open probability. Molecular dynamics simulations corroborate a gate-opening mechanism driven by inter-subunit hydrogen-bond networks. These findings provide a blueprint for designing subtype-selective drugs targeting synaptic transmission disorders."),
    ("Single-cell transcriptomics of pancreatic beta-cell heterogeneity in type 2 diabetes",
     "Using droplet-based single-cell RNA sequencing we profile 80,000 islet cells from healthy donors and patients with type 2 diabetes. A rare beta-cell subpopulation expressing elevated levels of stress-response genes expands in diabetic donors. Trajectory analysis shows that chronic hyperglycaemia drives dedifferentiation towards a progenitor-like state, suggesting new targets for beta-cell regeneration therapies."),
    ("Mitochondrial fission-fusion dynamics regulate neuronal apoptosis",
     "We show that balanced mitochondrial network morphology is required to prevent cytochrome c release in cortical neurons subjected to oxidative stress. Genetic ablation of DRP1 triggers hyperfusion and paradoxically accelerates apoptosis through impaired mitophagy. Pharmacological inhibition of MFN2 partially rescues network balance and reduces infarct volume in a rodent stroke model."),
    ("Gut microbiome composition predicts colorectal cancer immunotherapy response",
     "Shotgun metagenomic sequencing of pre-treatment faecal samples from 320 colorectal cancer patients treated with anti-PD-1 identifies Akkermansia muciniphila abundance as the strongest predictor of objective response. Germ-free mouse colonisation experiments confirm that A. muciniphila enhances tumour-infiltrating CD8-positive T cell density. 16S rRNA amplicon data replicate the association in an independent cohort."),
    ("Liquid-liquid phase separation of RNA-binding proteins in stress granule assembly",
     "We demonstrate that the low-complexity prion-like domains of TDP-43 and FUS undergo concentration-dependent liquid-liquid phase separation in vitro and in living cells under heat stress. Phosphorylation of serine residues within the LCD dissolves stress granules and restores nuclear localisation. Disease-associated mutations accelerate the transition from liquid to solid aggregates, linking phase behaviour to ALS pathology."),
    ("Epigenetic reprogramming during zebrafish embryo development",
     "Genome-wide ATAC-seq and CUT-and-RUN profiling across zebrafish embryogenesis reveals a wave of chromatin opening at 4 to 8 cell stage coinciding with zygotic genome activation. Pioneer transcription factors Pou5f3 and Sox19b cooperatively remodel nucleosome positioning at cis-regulatory elements. Maternal-to-zygotic transition timing is sensitive to temperature, with implications for fish adaptation to climate change."),
    ("Mechanosensitive ion channels in vascular endothelial flow sensing",
     "PIEZO1 mediates shear-stress-induced calcium influx in vascular endothelial cells, triggering nitric oxide release and vasodilation. CRISPR knockout of PIEZO1 elevates systemic blood pressure in mice under high-flow conditions. Structure-guided mutagenesis of the mechanosensitive gate identifies three residues critical for force transmission, offering targets for hypertension drug discovery."),
    ("Autophagy flux regulation by AMPK-mTORC1 signalling under nutrient deprivation",
     "We delineate a molecular switch in which AMPK phosphorylation of ULK1 at Ser317 and Ser777 activates autophagy initiation while concurrent mTORC1-mediated phosphorylation at Ser757 inhibits it. Live-cell imaging using a tandem fluorescent LC3 reporter quantifies autophagic flux in real time. Nutrient cycling experiments reveal oscillatory autophagy bursts that are dampened by oncogenic RAS signalling."),
    ("Adaptive immunity to SARS-CoV-2 Omicron sub-variants in vaccinated individuals",
     "Longitudinal analysis of T cell and B cell responses in triple-vaccinated donors exposed to BA.4 and BA.5 Omicron sub-variants shows that memory CD4-positive T cells cross-react broadly across spike epitopes despite antibody evasion. Single-cell BCR sequencing identifies convergent clonotypes with pan-sarbecovirus neutralisation capacity. Hybrid immunity from infection plus vaccination generates the widest neutralisation breadth."),
    ("Plant secondary metabolite biosynthesis under drought and salinity stress",
     "Transcriptome profiling of Arabidopsis thaliana under combined drought and salinity stress reveals upregulation of the phenylpropanoid and glucosinolate pathways. Metabolomics confirms accumulation of sinapic acid esters and aliphatic glucosinolates that correlate negatively with reactive oxygen species levels. Overexpression of MYB transcription factors recapitulates the metabolic response and improves survival under combined stress."),
    ("Spatial transcriptomics of tumour microenvironment heterogeneity in breast cancer",
     "Visium spatial transcriptomics applied to 24 treatment-naive breast tumours identifies spatially distinct niches harbouring immunosuppressive macrophage sub-states adjacent to cancer stem-like cells. Neighbourhood analysis reveals that proximity between CXCL10-expressing fibroblasts and cytotoxic T cells predicts pathological complete response to neoadjuvant chemotherapy. Integration with bulk RNA-seq deconvolution validates cell-type abundance estimates."),
    ("Telomere shortening and cellular senescence in ageing skeletal muscle",
     "We quantify telomere length by quantitative FISH across satellite cell populations in human muscle biopsies spanning ages 20 to 85. Critically short telomeres accumulate in satellite cells from donors over 60 and correlate with p21 induction and senescence-associated secretory phenotype markers. Exercise training partially reverses telomere attrition in middle-aged donors, suggesting an epigenetic mechanism linking physical activity to muscle regenerative capacity."),
    ("Wnt-Notch crosstalk in intestinal stem cell niche specification",
     "Organoid co-culture experiments and in vivo mouse genetic models demonstrate that Wnt3a secreted by Paneth cells and DLL4-mediated Notch signalling act antagonistically to specify absorptive enterocytes versus secretory goblet cell lineages. Single-cell multiome sequencing captures chromatin and transcriptional states at branching decision points. Pharmacological inhibition of gamma-secretase skews differentiation toward goblet cells, relevant to inflammatory bowel disease therapy."),
    ("Optogenetic dissection of hippocampal memory engram reactivation",
     "Channelrhodopsin-2 is expressed in memory-encoding CA1 neurons via activity-dependent TRAP2 mice to label contextual fear memory engrams. Light-induced reactivation of engram cells is sufficient to drive fear recall in a neutral context. Inhibitory opsin silencing of engram cells during recall impairs extinction, demonstrating that engram reactivation is necessary and sufficient for memory expression. Calcium imaging reveals a sparse, stable population code."),
    ("Structural and functional characterisation of plant NLR immune receptors",
     "Cryo-EM structures of the TIR-NLR protein RPP1 in its inactive monomeric and active resistosome states reveal a nucleotide-dependent oligomerisation mechanism. The resistosome acts as a NAD-cleaving enzyme that depletes cellular NAD and triggers hypersensitive cell death. AlphaFold2-guided mutagenesis identifies surface residues critical for effector recognition, providing a framework for engineering broad-spectrum disease resistance in crops."),
    ("Adipose tissue macrophage polarisation in obesity-induced insulin resistance",
     "Single-cell RNA sequencing of stromal vascular fraction from lean and obese mouse adipose tissue identifies a pro-inflammatory macrophage subset characterised by TREM2 and SPP1 expression that expands with high-fat diet. Selective depletion of TREM2-positive macrophages using conditional knockout improves whole-body glucose homeostasis. Spatial transcriptomics localises these cells to crown-like structures surrounding dying adipocytes."),
    ("Deep mutational scanning of influenza haemagglutinin receptor-binding domain",
     "We perform deep mutational scanning of all single amino acid substitutions across the H3 haemagglutinin receptor-binding domain using a yeast-display platform. Fitness landscapes reveal that positions 183, 226, and 228 tolerate only conservative substitutions without loss of sialic acid binding. Antigenic cartography integrating deep mutational scanning data and serology predicts immune escape trajectories relevant to seasonal vaccine strain selection."),
    ("Crosstalk between circadian clock and DNA damage response pathways",
     "The core circadian clock component BMAL1 directly interacts with PARP1 at sites of double-strand breaks, modulating repair pathway choice between NHEJ and homologous recombination in a time-of-day-dependent manner. Fibroblasts harvested at ZT0 show 40 percent higher HR efficiency than ZT12 cells. In vivo irradiation of mice at different circadian phases recapitulates this repair bias, with implications for chronotherapy scheduling in cancer radiotherapy."),
    ("Comparative genomics of antibiotic resistance evolution in Klebsiella pneumoniae",
     "Whole-genome sequencing of 450 Klebsiella pneumoniae clinical isolates across ten European hospitals identifies convergent acquisition of blaKPC-3 carbapenemase plasmids through horizontal gene transfer events traceable by phylogenomic reconstruction. Fitness cost analysis shows that compensatory mutations in the RNA polymerase beta subunit restore growth rate in carbapenem-resistant strains. Pan-genome analysis reveals a conserved accessory genome enriched in mobile genetic elements facilitating resistance spread."),
]

def make_template_csv(include_text: bool = True) -> str:
    names = [
        "Alice Hartley", "Ben Okafor", "Carmen Reyes", "David Chen", "Elena Vasquez",
        "Finn O'Brien", "Grace Kim", "Hassan Ali", "Iris Johansson", "James Patel",
        "Keiko Tanaka", "Luca Ferrari", "Maya Gupta", "Noel Adeyemi", "Olivia Muller",
        "Pedro Santos", "Quinn Walsh", "Rania Ibrahim", "Samuel Park", "Tara Novak",
    ]
    conflicts = [""] * 20
    conflicts[0]  = "Ben Okafor"
    conflicts[1]  = "Alice Hartley"
    conflicts[6]  = "Hassan Ali"
    conflicts[7]  = "Grace Kim"
    conflicts[14] = "Pedro Santos"
    conflicts[15] = "Olivia Muller"

    data = {
        "ID":                list(range(1, 21)),
        "ApplicantName":     names,
        "Email":             [f"{n.split()[0].lower()}.{n.split()[1].lower()}@university.ac.uk" for n in names],
        "ApplicationNumber": [f"APP{str(i).zfill(3)}" for i in range(1, 21)],
        "Conflicts":         conflicts,
    }
    if include_text:
        data["ApplicationText"] = [ab[1] for ab in BIOLOGY_ABSTRACTS]

    return pd.DataFrame(data).to_csv(index=False)


# ═══════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    mode_label = st.radio(
        "Allocation mode",
        ["🎲 Random", "🔍 TF-IDF similarity", "🧠 Embeddings (best quality)"],
    )
    if "Embeddings" in mode_label:
        mode = "embeddings"
    elif "TF-IDF" in mode_label:
        mode = "semantic"
    else:
        mode = "random"

    st.markdown("---")

    reviews_per_person = st.number_input(
        "Reviews per person", min_value=1, max_value=30, value=5,
        help="How many applications each person reviews (and receives reviews from)."
    )

    if mode in ("semantic", "embeddings"):
        floor_pct = st.slider(
            "Similarity floor (percentile)", min_value=0, max_value=50, value=20,
            help="Pairs below this percentile of overall similarity scores trigger a warning."
        )
    else:
        floor_pct = 20

    seed_col, info_col = st.columns([5, 1])
    with seed_col:
        random_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=122)
    with info_col:
        st.markdown("<div style='margin-top:1.9rem'></div>", unsafe_allow_html=True)
        with st.popover("ℹ️"):
            st.markdown("""
**What is a random seed?**

The allocation involves random choices. The seed is a "starting point" for that randomness:

- **Same seed = same result every time** — useful for reproducibility.
- **Different seed = different allocation** — change it to explore alternatives.

The default `122` matches the original R script.
            """)

    st.markdown("---")
    st.markdown("### 📋 Required CSV columns")
    for col, desc in [
        ("**ID**", "Unique integer"),
        ("**ApplicantName**", "Full name"),
        ("**Email**", "Email address"),
        ("**ApplicationNumber**", "Reference code"),
        ("**Conflicts**", "Comma-separated conflict names (can be empty)"),
    ]:
        st.markdown(f"{col} — {desc}")

    if mode in ("semantic", "embeddings"):
        st.markdown("**ApplicationText** — Full application text *(required for TF-IDF / Embeddings mode)*")

    st.markdown("---")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇️ Template\n(with text)",
            data=make_template_csv(include_text=True),
            file_name="dpr_template_semantic.csv",
            mime="text/csv",
        )
    with dl_col2:
        st.download_button(
            "⬇️ Template\n(basic)",
            data=make_template_csv(include_text=False),
            file_name="dpr_template_basic.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════

st.markdown('<p class="hero-title">Distributed Peer Review<br>Allocator 🔬</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload your applicants, choose your allocation mode, and generate a fair reviewer assignment in seconds.</p>', unsafe_allow_html=True)

with st.expander("ℹ️  How does this work?", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Step 1", "Split the pool",      "All applicants are randomly split into two equal groups."),
        ("Step 2", "Cross-group review",  "Group 1 reviews Group 2 and vice versa — no one reviews their own group."),
        ("Step 3", "Respect constraints", "Declared conflicts and reciprocal reviews are never paired together."),
        ("Step 4", "Optimise matching",   "In semantic mode, reviewers are matched to the most topically similar applications using TF-IDF cosine similarity."),
    ]
    for col, (num, title, desc) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <span class="step-number">{num}</span>
                <p class="step-title">{title}</p>
                <p style="font-size:0.88rem;color:#555;margin-top:0.4rem">{desc}</p>
            </div>""", unsafe_allow_html=True)

    if mode == "semantic":
        st.info("""
**About TF-IDF matching**

TF-IDF represents each application as a weighted word-count vector, up-weighting rare distinctive
terms and down-weighting common ones. Fast and requires no extra model download, but works purely
on word overlap — it cannot understand meaning or synonyms. Works best with longer texts.
A similarity score of 1.0 means identical vocabulary; 0.0 means no shared distinctive terms.
        """)
    elif mode == "embeddings":
        st.info("""
**About Embeddings matching**

Uses the `all-MiniLM-L6-v2` sentence-transformer model (~90 MB, downloaded once and cached).
Each application is encoded into a 384-dimension vector that captures *meaning*, not just word overlap —
so "monetary policy" and "central bank interest rates" score as similar even with no shared words.
This gives substantially better matching quality than TF-IDF, especially for short abstracts.
The model runs locally — no API key or cost per run.
        """)

st.markdown("---")
mode_labels = {
    "random":     "🎲 Random allocation",
    "semantic":   "🔍 TF-IDF semantic matching",
    "embeddings": "🧠 Embeddings semantic matching (sentence-transformers)",
}
st.markdown(f"**Mode:** {mode_labels[mode]}")
st.markdown("---")

# ── File upload ──
st.subheader("📂 Upload your applicant data")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    try:
        master_df = pd.read_csv(uploaded_file)

        required_cols = {"ID", "ApplicantName", "Email"}
        if mode in ("semantic", "embeddings"):
            required_cols.add("ApplicationText")

        missing = required_cols - set(master_df.columns)
        if missing:
            st.error(f"❌ Your CSV is missing these required columns: **{', '.join(sorted(missing))}**")
            st.stop()

        if "ApplicationNumber" not in master_df.columns:
            master_df["ApplicationNumber"] = master_df["ID"]
        if "Conflicts" not in master_df.columns:
            master_df["Conflicts"] = ""

        master_df["ID"] = master_df["ID"].astype(int)

        if mode in ("semantic", "embeddings"):
            empty_text = master_df["ApplicationText"].fillna("").str.strip().eq("").sum()
            if empty_text > 0:
                st.warning(f"⚠️ {empty_text} applicant(s) have empty ApplicationText — they will be matched less accurately.")

        st.markdown("**Preview of uploaded data:**")
        preview_cols = [c for c in master_df.columns if c != "ApplicationText"]
        st.dataframe(master_df[preview_cols].head(10), use_container_width=True)

        n = len(master_df)
        min_needed = reviews_per_person * 2 + 1
        if n < min_needed:
            st.warning(
                f"⚠️ You have **{n} applicants** but need at least **{min_needed}** "
                f"for everyone to receive {reviews_per_person} reviews. "
                "Consider reducing *Reviews per person* in the sidebar."
            )

        st.markdown(f"✅ **{n} applicants** loaded successfully.")
        st.markdown("---")

        if st.button("🚀 Run Allocation", type="primary"):
            with st.spinner("Running allocation…"):
                try:
                    df_combined, df_g1, df_g2, stats = run_allocation(
                        master_df.copy(), reviews_per_person, int(random_seed),
                        mode=mode, floor_percentile=floor_pct,
                    )

                    st.success("✅ Allocation complete!")
                    st.markdown("---")

                    # ── Basic metrics ──
                    st.subheader("📊 Results summary")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total applicants", stats["total_applicants"])
                    c2.metric("Group 1 size",     stats["group1_size"])
                    c3.metric("Group 2 size",     stats["group2_size"])
                    c4.metric("Got full reviews", stats["full_reviews"])

                    dist = stats["review_count_distribution"]
                    if dist:
                        vals = list(dist.values())
                        cA, cB, cC = st.columns(3)
                        cA.metric("Min reviews received", min(vals),
                                  delta=f"{min(vals) - reviews_per_person} vs target",
                                  delta_color="normal")
                        cB.metric("Avg reviews received", f"{np.mean(vals):.1f}")
                        cC.metric("Max reviews received", max(vals))

                    if stats["partial_reviews"] > 0:
                        st.warning(
                            f"⚠️ **{stats['partial_reviews']} applicant(s)** received fewer than "
                            f"{reviews_per_person} reviews. Try a different seed or check conflict constraints."
                        )

                    # ── Semantic quality panel ──
                    quality = stats.get("quality", {})
                    if quality and mode in ("semantic", "embeddings"):
                        st.markdown("---")
                        st.subheader("🎯 Semantic matching quality")

                        overall_mean = quality["overall_mean"]
                        overall_min  = quality["overall_min"]

                        qc1, qc2 = st.columns(2)
                        qc1.metric("Mean similarity score", f"{overall_mean:.4f}",
                                   help="Average TF-IDF cosine similarity across all reviewer–reviewee pairs. Higher = more topically aligned.")
                        qc2.metric("Min similarity score",  f"{overall_min:.4f}",
                                   help="The lowest similarity score in the entire allocation.")

                        sim_matrix = quality["sim_matrix"]
                        all_sims   = sim_matrix[sim_matrix > 0].flatten()
                        floor_val  = float(np.percentile(all_sims, floor_pct)) if len(all_sims) else 0.0

                        if overall_min < floor_val:
                            st.warning(
                                f"⚠️ The lowest pair similarity ({overall_min:.4f}) is below the "
                                f"{floor_pct}th percentile floor ({floor_val:.4f}). "
                                "Some reviewers may be poorly matched. Try a different seed or relax conflict constraints."
                            )
                        else:
                            st.success(f"✅ All pairs are above the {floor_pct}th percentile similarity floor ({floor_val:.4f}).")

                        with st.expander("🗺️ View similarity matrix (up to 20 applicants)", expanded=False):
                            ids_list   = quality["ids"]
                            names_list = quality["names"]
                            id_to_idx  = quality["id_to_idx"]
                            display_n  = min(20, len(ids_list))
                            sub_ids    = ids_list[:display_n]
                            sub_names  = [n.split()[0] for n in names_list[:display_n]]
                            idxs       = [id_to_idx[i] for i in sub_ids]
                            sub_sim    = sim_matrix[np.ix_(idxs, idxs)]
                            sim_df     = pd.DataFrame(sub_sim, index=sub_names, columns=sub_names).round(3)
                            st.dataframe(
                                sim_df.style.background_gradient(cmap="YlOrRd", vmin=0, vmax=1),
                                use_container_width=True,
                            )
                            st.caption("Cosine similarity between application texts. Diagonal is 0 (self-similarity suppressed). Warmer colour = more similar.")

                    # ── Preview table ──
                    st.markdown("---")
                    st.subheader("🗂️ Allocation preview (first 20 rows)")
                    preview_cols = (
                        ["Reviewer_Name", "Reviewer_Email", "Reviewer_ApplicationNumber"]
                        + [c for c in df_combined.columns if c.startswith("Reviewee_") and "_Name" in c]
                    )
                    st.dataframe(df_combined[preview_cols].head(20), use_container_width=True)

                    # ── Downloads ──
                    st.markdown("---")
                    st.subheader("⬇️ Download results")

                    def to_csv(df):
                        return df.to_csv(index=False).encode("utf-8")

                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr("dpr_all_allocations.csv",    df_combined.to_csv(index=False))
                        zf.writestr("dpr_group1_allocations.csv", df_g1.to_csv(index=False))
                        zf.writestr("dpr_group2_allocations.csv", df_g2.to_csv(index=False))
                        if quality and mode in ("semantic", "embeddings"):
                            ids_list   = quality["ids"]
                            names_list = quality["names"]
                            id_to_idx  = quality["id_to_idx"]
                            sim_matrix = quality["sim_matrix"]
                            labels = [f"{i}_{n}" for i, n in zip(ids_list, names_list)]
                            sim_df_full = pd.DataFrame(sim_matrix, index=labels, columns=labels).round(4)
                            zf.writestr("dpr_similarity_matrix.csv", sim_df_full.to_csv())
                    zip_buf.seek(0)

                    cd, ce, cf = st.columns(3)
                    with cd:
                        st.download_button("📦 Download all (ZIP)", data=zip_buf.getvalue(),
                                           file_name="dpr_results.zip", mime="application/zip")
                    with ce:
                        st.download_button("📄 Group 1 CSV", data=to_csv(df_g1),
                                           file_name="dpr_group1_allocations.csv", mime="text/csv")
                    with cf:
                        st.download_button("📄 Group 2 CSV", data=to_csv(df_g2),
                                           file_name="dpr_group2_allocations.csv", mime="text/csv")

                    note = "Each CSV row is one reviewer with their assigned reviewees."
                    if mode in ("semantic", "embeddings"):
                        note += " Similarity scores per pair are included. The ZIP also contains the full similarity matrix."
                    st.caption(note)

                except Exception as e:
                    st.error(f"❌ Allocation failed: {e}")
                    st.exception(e)

    except Exception as e:
        st.error(f"❌ Could not read your CSV: {e}")

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;background:#f7f7fb;border-radius:12px;border:2px dashed #c7c7e0;">
        <span style="font-size:3rem">📎</span>
        <p style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#1a1a2e;margin:0.5rem 0">
            Drop your CSV here to get started
        </p>
        <p style="color:#888;font-size:0.9rem;">
            Download a template from the sidebar — the <em>with text</em> version includes 20 biology abstracts ready to test semantic matching.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='font-size:0.78rem;color:#aaa;text-align:center;'>"
    "Based on the DPR allocation algorithm by Cillian Brophy · "
    "Random and semantic (TF-IDF) modes · Conflicts and reciprocals respected."
    "</p>",
    unsafe_allow_html=True,
)
