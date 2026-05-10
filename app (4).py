"""
app.py — Home / landing page of the DPR Allocator.
Navigate to the Allocator page via the sidebar to run an allocation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from dpr_core import inject_css

st.set_page_config(
    page_title="DPR Allocator — Home",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

st.markdown('<p class="hero-title">Distributed Peer Review<br>Allocator 🔬</p>', unsafe_allow_html=True)
st.markdown("""
<p class="hero-sub">
A free tool for fairly assigning reviewers to applications — with optional AI-powered topic matching.<br>
Built on the algorithm by <strong>Cillian Brophy</strong>. Use the sidebar to navigate to the Allocator.
</p>
""", unsafe_allow_html=True)
st.markdown("---")

st.subheader("What is Distributed Peer Review?")
st.markdown("""
Distributed Peer Review (DPR) is a method used by funding bodies, journals, and academic programmes
to fairly spread the burden of reviewing applications. Instead of a small panel reading everything,
**every applicant also acts as a reviewer** — reading and scoring a subset of their peers' work.

This has several advantages:

- The reviewing load is spread across the whole pool, not just a few people
- Reviewers have relevant expertise — they applied to the same scheme
- It scales well — more applicants means more reviewers
- Conflicts of interest are systematically excluded
""")
st.markdown("---")

st.subheader("How this tool works")
col1, col2, col3, col4 = st.columns(4)
steps = [
    ("1", "Upload your data",   "Provide a CSV with applicant names, emails, and optionally the full text of each application."),
    ("2", "Choose a mode",      "Pick random, TF-IDF, or AI embeddings matching depending on how sophisticated you want the pairing to be."),
    ("3", "Run the allocation", "The algorithm splits applicants into two groups, assigns reviewers, respects conflicts, and prevents reciprocal reviews."),
    ("4", "Download results",   "Get CSV files listing each person's assigned reviewees, ready to send out."),
]
for col, (num, title, desc) in zip([col1, col2, col3, col4], steps):
    with col:
        st.markdown(f"""
        <div class="step-card">
            <span class="step-number">Step {num}</span>
            <p class="step-title">{title}</p>
            <p style="font-size:0.88rem;color:#555;margin-top:0.4rem">{desc}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Choosing an allocation mode")
mc1, mc2, mc3 = st.columns(3)

with mc1:
    st.markdown("""
    <div class="info-card">
        <h4>🎲 Random allocation</h4>
        <p>Reviewers are assigned randomly, subject only to conflict rules and
        the rule that no one reviews someone who is reviewing them back.<br><br>
        <strong>Best for:</strong> pools where all applications are in the same
        field, or where you have no application text.<br><br>
        <strong>No text required.</strong></p>
    </div>""", unsafe_allow_html=True)

with mc2:
    st.markdown("""
    <div class="info-card">
        <h4>🔍 TF-IDF matching</h4>
        <p>Compares application texts by counting shared distinctive words.
        Applications using similar vocabulary are paired together.<br><br>
        <strong>Best for:</strong> a quick improvement over random when you have
        texts but do not need high precision.<br><br>
        <strong>Limitation:</strong> no understanding of meaning. "Monetary policy"
        and "central bank rates" would score as unrelated.</p>
    </div>""", unsafe_allow_html=True)

with mc3:
    st.markdown("""
    <div class="info-card">
        <h4>🧠 Embeddings matching</h4>
        <p>Uses a small AI language model to encode each application into a
        rich vector that captures meaning, not just word overlap.
        Substantially better matching quality than TF-IDF.<br><br>
        <strong>Best for:</strong> multi-disciplinary pools where reviewers
        should be matched by topic expertise.<br><br>
        <strong>First run only:</strong> downloads a 90 MB model, then caches it.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Practical limits and guidance")
lc1, lc2 = st.columns(2)

with lc1:
    st.markdown("""
    <div class="warn-card">
        <h4>⚠️ Text length limit (embeddings mode)</h4>
        <p>The AI model processes a maximum of approximately <strong>200 words (256 tokens)</strong>
        per application. Text beyond this is silently ignored.<br><br>
        <strong>Recommendation:</strong> ask applicants for a structured abstract of
        150 to 200 words covering background, aims, methods, and expected outcomes.
        This fits the model perfectly and is more useful for reviewers too.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-card">
        <h4>⚠️ Minimum applicant numbers</h4>
        <p>The algorithm needs at least <strong>(Reviews per person x 2) + 1</strong>
        applicants. For the default of 5 reviews per person, that means at least
        11 applicants.<br><br>
        With heavy conflict constraints you may need more. The app will warn you
        if the pool is too small.</p>
    </div>""", unsafe_allow_html=True)

with lc2:
    st.markdown("""
    <div class="info-card">
        <h4>📊 How many applicants can this handle?</h4>
        <p><strong>Random and TF-IDF:</strong> scales to thousands with no issues.<br><br>
        <strong>Embeddings:</strong> the model uses approximately 90 MB of memory (fixed cost).
        Storing embeddings for 500 applicants adds less than 1 MB.
        Encoding 500 texts takes roughly 30 to 60 seconds on a shared CPU.<br><br>
        <strong>Practical ceiling:</strong> around 500 to 1000 applicants on the free
        Streamlit Cloud tier. Beyond that, consider self-hosting.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>🔒 Does this store my data?</h4>
        <p>No. Uploaded data lives only in server memory for the duration of
        your session and is discarded when you close the browser.
        Nothing is written to a database or disk.<br><br>
        For GDPR compliance with real applicant data, consider anonymising your
        CSV by using IDs instead of names and emails, or run the app locally on
        your own machine so data never leaves your computer.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Your CSV file format")
st.markdown("The app expects a CSV with these columns:")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
| Column | Required? | Description |
|---|---|---|
| `ID` | Yes | Unique integer per applicant |
| `ApplicantName` | Yes | Full name |
| `Email` | Yes | Email address |
| `ApplicationNumber` | Optional | Reference code e.g. APP001 |
| `Conflicts` | Optional | Comma-separated names of people this person cannot review |
| `ApplicationText` | Semantic modes | Application text, 150 to 200 words recommended |
""")
with col_b:
    st.info("""
**Conflicts column format**

List full names of anyone this person must not review, comma-separated:

`Alice Smith, Bob Jones`

Leave blank if no conflicts. The algorithm also automatically prevents A reviewing B if B is already reviewing A.
    """)

st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:2rem 1rem;background:linear-gradient(135deg,#f0f0ff,#faf0ff);
border-radius:12px;border:1px solid #dcdcf5;margin-bottom:1rem;">
    <p style="font-family:'DM Serif Display',serif;font-size:1.6rem;color:#1a1a2e;margin:0 0 0.5rem 0;">
        Ready to run an allocation?
    </p>
    <p style="color:#555;margin:0;">
        Head to the <strong>Allocator</strong> page in the left sidebar
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='font-size:0.78rem;color:#aaa;text-align:center;'>"
    "Based on the DPR allocation algorithm by Cillian Brophy · "
    "Random, TF-IDF, and Embeddings modes · Free to use · No data stored."
    "</p>",
    unsafe_allow_html=True,
)
