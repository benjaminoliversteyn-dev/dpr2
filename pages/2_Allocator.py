"""
pages/2_Allocator.py — The main allocation tool page.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import io
import zipfile
import streamlit as st
import pandas as pd
import numpy as np

from dpr_core import (
    inject_css, run_allocation, make_template_csv,
)

st.set_page_config(
    page_title="DPR Allocator — Run",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

st.markdown('<p class="hero-title">Run Allocation 🚀</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload your CSV, configure settings in the sidebar, and generate your reviewer assignments.</p>', unsafe_allow_html=True)
st.markdown("---")

# ═══════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Settings")
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
        help="How many applications each person reviews and receives reviews from."
    )

    if mode in ("semantic", "embeddings"):
        floor_pct = st.slider(
            "Similarity floor (percentile)", min_value=0, max_value=50, value=20,
            help="Pairs below this percentile of all similarity scores trigger a warning."
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

The seed is a starting point for all random choices in the allocation.

- **Same seed = same result every time** — useful for reproducibility.
- **Different seed = different allocation** — change it to explore alternatives.

The default 122 matches the original R script.
            """)

    st.markdown("---")
    st.markdown("### Required CSV columns")
    for col, desc in [
        ("`ID`", "Unique integer"),
        ("`ApplicantName`", "Full name"),
        ("`Email`", "Email address"),
        ("`ApplicationNumber`", "Reference code (optional)"),
        ("`Conflicts`", "Comma-separated conflict names (optional)"),
    ]:
        st.markdown(f"{col} — {desc}")
    if mode in ("semantic", "embeddings"):
        st.markdown("`ApplicationText` — Application text, 150–200 words *(required)*")

    st.markdown("---")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Template\n(with text)",
            data=make_template_csv(include_text=True),
            file_name="dpr_template_semantic.csv",
            mime="text/csv",
        )
    with dl2:
        st.download_button(
            "⬇️ Template\n(basic)",
            data=make_template_csv(include_text=False),
            file_name="dpr_template_basic.csv",
            mime="text/csv",
        )

# ═══════════════════════════════════════════════════════
#  MODE INFO BANNER
# ═══════════════════════════════════════════════════════
mode_labels = {
    "random":     "🎲 Random allocation",
    "semantic":   "🔍 TF-IDF semantic matching",
    "embeddings": "🧠 Embeddings semantic matching (sentence-transformers)",
}
st.markdown(f"**Mode:** {mode_labels[mode]}")

if mode == "semantic":
    st.info("TF-IDF matches applications by shared distinctive vocabulary. Fast, no model download needed. Works best with longer texts (200+ words). Cannot understand meaning — synonyms will not match.")
elif mode == "embeddings":
    st.info("Embeddings mode uses a small AI model (all-MiniLM-L6-v2, ~90 MB) to understand meaning rather than just word overlap. Best quality matching. The model is downloaded once and cached. Text is capped at ~200 words — longer text is silently truncated.")

st.markdown("---")

# ═══════════════════════════════════════════════════════
#  FILE UPLOAD
# ═══════════════════════════════════════════════════════
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
            st.error(f"Your CSV is missing these required columns: **{', '.join(sorted(missing))}**")
            st.stop()

        if "ApplicationNumber" not in master_df.columns:
            master_df["ApplicationNumber"] = master_df["ID"]
        if "Conflicts" not in master_df.columns:
            master_df["Conflicts"] = ""

        master_df["ID"] = master_df["ID"].astype(int)

        if mode in ("semantic", "embeddings"):
            empty_text = master_df["ApplicationText"].fillna("").str.strip().eq("").sum()
            if empty_text > 0:
                st.warning(f"{empty_text} applicant(s) have empty ApplicationText — they will be matched less accurately.")

        st.markdown("**Preview (metadata only):**")
        preview_cols = [c for c in master_df.columns if c != "ApplicationText"]
        st.dataframe(master_df[preview_cols].head(10), use_container_width=True)

        n = len(master_df)
        min_needed = reviews_per_person * 2 + 1
        if n < min_needed:
            st.warning(
                f"You have **{n} applicants** but need at least **{min_needed}** "
                f"for everyone to receive {reviews_per_person} reviews. "
                "Consider reducing Reviews per person in the sidebar."
            )

        st.markdown(f"**{n} applicants loaded.**")
        st.markdown("---")

        if st.button("🚀 Run Allocation", type="primary"):
            with st.spinner("Running allocation…"):
                try:
                    df_combined, df_g1, df_g2, stats = run_allocation(
                        master_df.copy(), reviews_per_person, int(random_seed),
                        mode=mode, floor_percentile=floor_pct,
                    )

                    st.success("Allocation complete!")
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
                            f"**{stats['partial_reviews']} applicant(s)** received fewer than "
                            f"{reviews_per_person} reviews. Try a different seed or check conflict constraints."
                        )

                    # ── Semantic quality panel ──
                    quality = stats.get("quality", {})
                    if quality and mode in ("semantic", "embeddings"):
                        st.markdown("---")
                        st.subheader("🎯 Matching quality")

                        overall_mean = quality["overall_mean"]
                        overall_min  = quality["overall_min"]
                        qc1, qc2 = st.columns(2)
                        qc1.metric("Mean similarity score", f"{overall_mean:.4f}",
                                   help="Average cosine similarity across all reviewer-reviewee pairs. Higher = more topically aligned.")
                        qc2.metric("Min similarity score", f"{overall_min:.4f}",
                                   help="The lowest similarity score in the entire allocation.")

                        sim_matrix = quality["sim_matrix"]
                        all_sims   = sim_matrix[sim_matrix > 0].flatten()
                        floor_val  = float(np.percentile(all_sims, floor_pct)) if len(all_sims) else 0.0

                        if overall_min < floor_val:
                            st.warning(
                                f"The lowest pair similarity ({overall_min:.4f}) is below the "
                                f"{floor_pct}th percentile floor ({floor_val:.4f}). "
                                "Some reviewers may be poorly matched. Try a different seed or relax conflict constraints."
                            )
                        else:
                            st.success(f"All pairs are above the {floor_pct}th percentile similarity floor ({floor_val:.4f}).")

                        with st.expander("View similarity matrix (up to 20 applicants)", expanded=False):
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
                            st.caption("Cosine similarity between application texts. Diagonal is 0. Warmer = more similar.")

                    # ── Preview ──
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
                    st.error(f"Allocation failed: {e}")
                    st.exception(e)

    except Exception as e:
        st.error(f"Could not read your CSV: {e}")

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;background:#f7f7fb;border-radius:12px;border:2px dashed #c7c7e0;">
        <span style="font-size:3rem">📎</span>
        <p style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#1a1a2e;margin:0.5rem 0">
            Drop your CSV here to get started
        </p>
        <p style="color:#888;font-size:0.9rem;">
            Download a template from the sidebar. The <em>with text</em> version includes 30 biology,
            economics, and philosophy abstracts — perfect for testing semantic matching.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='font-size:0.78rem;color:#aaa;text-align:center;'>"
    "Based on the DPR allocation algorithm by Cillian Brophy · "
    "Random, TF-IDF, and Embeddings modes · Conflicts and reciprocals respected."
    "</p>",
    unsafe_allow_html=True,
)
