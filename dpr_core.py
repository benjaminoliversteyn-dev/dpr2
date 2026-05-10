"""
dpr_core.py — shared backend logic for the DPR Allocator.
Imported by all pages so allocation code lives in one place.
"""

import re
import random
from collections import Counter
from math import log

import numpy as np
import pandas as pd
import streamlit as st


# ═══════════════════════════════════════════════════════
#  SHARED CSS
# ═══════════════════════════════════════════════════════

CSS = """
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
.info-card {
    background: #f0f0ff; border-radius: 10px;
    padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    border: 1px solid #dcdcf5;
}
.info-card h4 { margin: 0 0 0.4rem 0; color: #1a1a2e; font-size: 1rem; }
.info-card p  { margin: 0; color: #555; font-size: 0.9rem; line-height: 1.5; }
.warn-card {
    background: #fffbeb; border-radius: 10px;
    padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    border: 1px solid #fcd34d;
}
.warn-card h4 { margin: 0 0 0.4rem 0; color: #92400e; font-size: 1rem; }
.warn-card p  { margin: 0; color: #78350f; font-size: 0.9rem; line-height: 1.5; }
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
"""

def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  TF-IDF SIMILARITY ENGINE
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


def cosine_similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, 0.0)
    return sim


# ═══════════════════════════════════════════════════════
#  SENTENCE-EMBEDDINGS ENGINE
# ═══════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading embedding model (first run only — ~90 MB download)…")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_embeddings_matrix(corpus: list) -> np.ndarray:
    model = load_embedding_model()
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


# ═══════════════════════════════════════════════════════
#  ALLOCATION LOGIC
# ═══════════════════════════════════════════════════════

def parse_conflicts(conflict_str) -> list:
    if pd.isna(conflict_str) or str(conflict_str).strip() == "":
        return []
    return [c.strip() for c in str(conflict_str).split(",") if c.strip()]


def generate_constraint_list(data1, data2, master) -> dict:
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


def dpr_allocation_semantic(group_to_allocate, group_from, constraint_list,
                             sim_matrix, id_to_idx, reviews_per_person=10):
    from_ids = list(group_from["ID"])
    count = {fid: 0 for fid in from_ids}
    assignments = {}

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
            candidates.append((fid, float(sim_matrix[r_idx, id_to_idx[fid]])))
        candidates.sort(key=lambda x: -x[1])
        chosen = [fid for fid, _ in candidates[:reviews_per_person]]
        assignments[reviewer_id] = chosen
        for c in chosen:
            count[c] += 1
    return assignments, count


def redistribute_incomplete(assignments, group_to_allocate, group_from,
                             constraint_list, count, reviews_per_person=10,
                             sim_matrix=None, id_to_idx=None):
    from_ids = set(group_from["ID"])
    for _ in range(500):
        short = [rid for rid, lst in assignments.items() if len(lst) < reviews_per_person]
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


def build_final_df(assignments, master, reviews_per_person, sim_matrix=None, id_to_idx=None):
    id_to_name   = dict(zip(master["ID"], master["ApplicantName"]))
    id_to_email  = dict(zip(master["ID"], master.get("Email", pd.Series(dtype=str))))
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
                row[f"Reviewee_{i}_Similarity"] = round(
                    float(sim_matrix[id_to_idx[reviewer_id], id_to_idx[rev_id]]), 4)
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
    return reviewer_means, (float(np.mean(scores)) if scores else 0.0), (float(np.min(scores)) if scores else 0.0)


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

    df_g1 = build_final_df(g1_assignments, master, reviews_per_person, sim_matrix, id_to_idx)
    df_g2 = build_final_df(g2_assignments, master, reviews_per_person, sim_matrix, id_to_idx)
    df_combined = pd.concat([df_g1, df_g2], ignore_index=True)

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
#  TEMPLATE CSV DATA
# ═══════════════════════════════════════════════════════

BIOLOGY_ABSTRACTS = [
    "This project investigates CRISPR-Cas9 gene editing in mammalian cells, focusing on off-target mutation rates and repair pathway selection. We will use whole-genome sequencing to characterise editing outcomes across diverse cell types and develop high-fidelity Cas9 variants that minimise unintended genomic damage while preserving therapeutic efficacy.",
    "We study mitochondrial dynamics in neurodegeneration, examining how fission and fusion balance affects cytochrome c release and apoptosis in Parkinson disease models. Pharmacological modulators of DRP1 and MFN2 will be tested in primary dopaminergic neurons and validated in rodent models of alpha-synuclein toxicity.",
    "Single-cell RNA sequencing of tumour-infiltrating immune cells will reveal the transcriptional states associated with checkpoint immunotherapy response in non-small cell lung cancer. We will integrate CITE-seq protein measurements with TCR clonotype data to identify exhausted T cell populations amenable to reinvigoration.",
    "This research characterises the gut microbiome-brain axis in major depressive disorder using shotgun metagenomics and metabolomics on stool and plasma samples from a longitudinal patient cohort. Germ-free mouse colonisation with patient-derived microbiota will establish causal links between microbial metabolites and hippocampal neurogenesis.",
    "We investigate liquid-liquid phase separation of FUS and TDP-43 in ALS pathogenesis, using in vitro reconstitution and optogenetic condensate induction in motor neurons. Cryo-electron tomography will resolve the ultrastructure of pathological aggregates to guide therapeutic disaggregation strategies.",
    "Epigenetic clocks based on DNA methylation will be calibrated and validated across diverse human tissues to quantify biological ageing rates. We will test whether lifestyle interventions including exercise and caloric restriction slow epigenetic ageing in a randomised controlled trial with six-month follow-up.",
    "Plant immunity signalling through NLR resistosome complexes will be structurally and biochemically characterised using cryo-EM and hydrogen-deuterium exchange mass spectrometry. Engineering of recognition specificity will be attempted to broaden disease resistance in wheat against multiple fungal pathogens.",
    "The circadian regulation of DNA double-strand break repair will be dissected using time-resolved ChIP-seq of repair factors across the 24-hour cycle in human fibroblasts. We will test whether chronotherapy scheduling of radiotherapy improves tumour control while reducing normal tissue toxicity in mouse xenograft models.",
    "Spatial transcriptomics of developing human cortex organoids will map the emergence of layer-specific neuronal identities and synaptic connectivity patterns. Comparison with primary foetal tissue will benchmark organoid fidelity and identify transcriptional programmes governing cortical folding in human evolution.",
    "We profile adaptive immune responses to seasonal influenza vaccination in elderly versus young adult donors using single-cell multiomics. Germinal centre dynamics reconstructed from B cell receptor lineage tracing will identify correlates of vaccine-induced broadly neutralising antibody generation across age groups.",
]

ECONOMICS_ABSTRACTS = [
    "This project estimates the causal effect of universal basic income on labour supply and entrepreneurship using a randomised controlled trial across three municipalities. Difference-in-differences analysis of administrative tax records will quantify heterogeneous treatment effects by pre-existing wealth, education, and local labour market conditions.",
    "We investigate how central bank forward guidance shapes household inflation expectations using a nationally representative survey experiment that randomises information provision. Structural VAR models estimated on high-frequency financial data will recover the transmission mechanism from monetary policy communication to real economic outcomes.",
    "The macroeconomic consequences of demographic ageing on savings rates, asset prices, and public pension sustainability will be quantified using an overlapping generations model calibrated to OECD panel data. We will simulate policy counterfactuals including retirement age reform and immigration to assess fiscal solvency under different scenarios.",
    "Using natural language processing applied to earnings call transcripts, we construct firm-level measures of climate transition risk exposure and link them to investment, employment, and credit spreads. Instrumental variable strategies exploiting regulatory shock dates will establish causal effects on corporate capital allocation.",
    "This research analyses optimal taxation of capital income in economies with heterogeneous agents and incomplete markets. New sufficient statistics for the welfare effects of tax reforms will be derived and empirically estimated using administrative wealth registry data from Scandinavian countries covering three decades.",
    "We study the economics of platform competition and two-sided markets using a structural demand model estimated on detailed transaction data from digital marketplaces. Counterfactual simulations will evaluate the welfare effects of interoperability mandates and data portability requirements under proposed EU digital regulation.",
    "The role of social networks in labour market matching will be estimated using linked employer-employee data merged with friendship network information from a large social media platform. Quasi-experimental variation in network structure from platform algorithm changes will identify peer effects on wages and occupational mobility.",
    "We examine how housing supply elasticity mediates the local employment effects of place-based policies such as enterprise zones and investment tax credits. Shift-share instruments for policy assignment combined with land use regulation indices will identify causal effects on local labour demand and commuting patterns.",
    "This project investigates the macroprudential effects of bank capital requirements on credit supply and economic growth using a difference-in-differences design around staggered Basel III implementation across jurisdictions. Loan-level data from credit registers will decompose aggregate lending responses into extensive and intensive margin adjustments.",
    "Using administrative records from a large developing economy, we study how digital financial inclusion affects household consumption smoothing, investment in children education, and resilience to idiosyncratic income shocks. Regression discontinuity designs exploiting eligibility thresholds in mobile money rollout will identify causal effects.",
]

PHILOSOPHY_ABSTRACTS = [
    "This project examines the metaphysics of personal identity across time, arguing against both psychological continuity and biological accounts in favour of a four-dimensionalist perdurance theory. We will analyse puzzle cases involving fission, gradual replacement, and teleportation to develop a more defensible criterion of diachronic identity.",
    "We investigate the epistemic foundations of scientific modelling, questioning whether idealised models with known false assumptions can genuinely explain empirical phenomena. A new account of explanatory fictions will be developed drawing on the philosophy of mathematics and the debate between scientific realism and instrumentalism.",
    "This research defends a neo-Kantian constructivist metaethics that grounds moral facts in the constitutive norms of rational agency rather than mind-independent moral properties. We will respond to expressivism, error theory, and robust realism, arguing that constructivism avoids the objections facing competing metaethical positions.",
    "We analyse the concept of epistemic injustice in institutional contexts, extending Miranda Fricker framework to cover structural forms of testimonial and hermeneutical injustice that cannot be reduced to individual acts of prejudice. Applications to medical diagnosis, legal testimony, and academic knowledge production will be developed.",
    "The philosophy of time will be examined through the lens of the growing block universe theory, which holds that past and present exist but the future does not. We will defend a novel version of this view that accommodates the special theory of relativity without collapsing into either eternalism or presentism.",
    "This project investigates whether artificial intelligence systems can possess morally relevant mental states such as consciousness, preferences, and suffering, drawing on philosophy of mind and empirical cognitive science. Practical implications for AI welfare and moral status will be developed with reference to utilitarian and Kantian ethical frameworks.",
    "We examine the relationship between testimony, trust, and knowledge, arguing that testimonial knowledge requires a relational account in which the epistemic standing of the speaker and the social conditions of assertion are constitutively relevant. The account will be applied to cases of expert disagreement and conspiracy theorising.",
    "This research investigates the ethics of existential risk, asking whether the potential extinction of humanity generates special moral obligations that override conventional constraints on individual rights. We will critically evaluate longtermist arguments and develop a pluralist account that balances future and present moral claims.",
    "We study the philosophy of causation in the biological sciences, examining whether mechanistic accounts of causation can accommodate the context-sensitivity and redundancy characteristic of gene regulatory networks. A dispositional account of biological causation will be defended against counterfactual and interventionist alternatives.",
    "This project examines free will and moral responsibility in light of recent neuroscientific findings on unconscious neural preparation before conscious intention. We will argue that compatibilist accounts of freedom grounded in reasons-responsiveness survive empirical challenges and ground a coherent practice of praise and blame.",
]


def make_template_csv(include_text: bool = True) -> str:
    names = [
        "Bella Hartley", "Bruno Okafor", "Beatrice Reyes", "Benedict Chen", "Bridget Vasquez",
        "Boris O'Brien", "Bianca Kim", "Barnaby Ali", "Bernadette Johansson", "Blaise Patel",
        "Eleanor Tanaka", "Edmund Ferrari", "Esme Gupta", "Ezra Adeyemi", "Edith Muller",
        "Eduardo Santos", "Elspeth Walsh", "Emeka Ibrahim", "Eugenia Park", "Ernst Novak",
        "Polly Krishnan", "Pascal Blanc", "Penelope Osei", "Piers Dupont", "Priya Mansour",
        "Petra McAllister", "Ptolemy Svensson", "Prudence Nakamura", "Phoenix Ibarra", "Paloma Wolff",
    ]
    conflicts = [""] * 30
    for a, b in [(0,1),(10,11),(20,21),(4,5),(14,15),(24,25)]:
        conflicts[a] = names[b]
        conflicts[b] = names[a]

    all_texts = BIOLOGY_ABSTRACTS + ECONOMICS_ABSTRACTS + PHILOSOPHY_ABSTRACTS

    data = {
        "ID":                list(range(1, 31)),
        "ApplicantName":     names,
        "Email":             [f"{n.split()[0].lower()}.{n.split()[-1].lower()}@uni.ac.uk" for n in names],
        "ApplicationNumber": [f"APP{str(i).zfill(3)}" for i in range(1, 31)],
        "Conflicts":         conflicts,
    }
    if include_text:
        data["ApplicationText"] = all_texts
    return pd.DataFrame(data).to_csv(index=False)
