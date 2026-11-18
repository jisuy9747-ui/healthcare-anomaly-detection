import streamlit as st

from src.ui import apply_theme


st.set_page_config(page_title="Home", page_icon="H", layout="wide")
apply_theme()

loaded = "raw_dataset" in st.session_state
processed = "processed_dataset" in st.session_state

st.markdown(
    """
    <section class="hero">
      <div class="status-chip"><span class="status-dot"></span>Research workspace</div>
      <h1>Healthcare Anomaly Detection</h1>
      <p>A complete comparative workflow for identifying healthcare records that
      deserve closer data-quality review using six unsupervised methods.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset status", "Loaded" if loaded else "Not loaded")
c2.metric("Records", f"{len(st.session_state['raw_dataset']):,}" if loaded else "0")
c3.metric("Selected features", len(st.session_state["processed_dataset"].columns) if processed else "0")
c4.metric("Algorithms available", "6")

st.markdown('<div class="section-heading">Recommended workflow</div>', unsafe_allow_html=True)
workflow = [
    ("1", "Upload & profile", "Import CSV or Excel data and inspect structure."),
    ("2", "Preprocess safely", "Select features, impute values and scale numerically."),
    ("3", "Detect anomalies", "Compare Isolation Forest, LOF, SVM, AE, KNN and DBSCAN."),
    ("4", "Review & export", "Inspect explanations, set review status and download reports."),
]
cards = st.columns(4)
for column, (step, title, description) in zip(cards, workflow):
    with column:
        st.markdown(
            f"""
            <div class="workflow-card">
              <div class="step">{step}</div>
              <h3>{title}</h3>
              <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-heading">Navigate to a stage</div>', unsafe_allow_html=True)
n1, n2, n3, n4 = st.columns(4)
with n1:
    st.page_link("pages/2_Dataset_Upload.py", label="Dataset upload", icon=":material/upload:", use_container_width=True)
with n2:
    st.page_link("pages/3_Preprocessing.py", label="Preprocessing", icon=":material/tune:", use_container_width=True)
with n3:
    st.page_link("pages/4_Anomaly_Detection.py", label="Run detection", icon=":material/neurology:", use_container_width=True)
with n4:
    st.page_link("pages/6_Algorithm_Comparison.py", label="Compare methods", icon=":material/analytics:", use_container_width=True)

st.markdown(
    """
    <div class="notice-card">
    <strong>Interpret with care:</strong> These methods identify statistical
    unusualness for investigation. They do not diagnose disease or automatically
    confirm a data error.
    </div>
    """,
    unsafe_allow_html=True,
)
