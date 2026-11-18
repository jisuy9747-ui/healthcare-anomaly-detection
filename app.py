import streamlit as st

from src.ui import apply_theme


st.set_page_config(
    page_title="Healthcare Anomaly Detection System",
    page_icon="H",
    layout="wide",
)

apply_theme()

loaded = "raw_dataset" in st.session_state
processed = "processed_dataset" in st.session_state

st.markdown(
    f"""
    <section class="hero">
      <div class="status-chip"><span class="status-dot"></span>
      {"Dataset loaded" if loaded else "Ready for a dataset"}
      </div>
      <h1>Healthcare Anomaly<br>Detection System</h1>
      <p>Compare six unsupervised learning methods and turn unusual healthcare
      records into review-ready evidence for responsible human investigation.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if loaded:
    st.caption("Current session data is preserved; preprocessing creates a separate copy.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset Status", "Loaded" if loaded else "Waiting")
col2.metric("Records", f"{len(st.session_state['raw_dataset']):,}" if loaded else "0")
col3.metric("Selected Features", len(st.session_state["processed_dataset"].columns) if processed else "0")
col4.metric("Algorithms", "6")

st.markdown('<div class="section-heading">Start your analysis</div>', unsafe_allow_html=True)
a, b, c = st.columns(3)
with a:
    st.page_link("pages/2_Dataset_Upload.py", label="1  Upload & inspect", icon=":material/upload:", use_container_width=True)
with b:
    st.page_link("pages/3_Preprocessing.py", label="2  Prepare features", icon=":material/tune:", use_container_width=True)
with c:
    st.page_link("pages/4_Anomaly_Detection.py", label="3  Run all methods", icon=":material/neurology:", use_container_width=True)

st.markdown(
    """
    <div class="notice-card">
    <strong>Responsible use:</strong> An anomaly is not a diagnosis or proof of
    an incorrect record. Every alert represents an item for qualified human review.
    </div>
    """,
    unsafe_allow_html=True,
)

