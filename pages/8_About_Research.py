import streamlit as st

from src.validation import check_leave_state_out


st.set_page_config(page_title="About the Research", page_icon="B", layout="wide")
st.title("About the Research")

st.subheader("Research problem")
st.write(
    "The project compares unsupervised anomaly-detection methods for unusual "
    "health records and potential data-quality problems. An anomaly is not "
    "a diagnosis or proof that a record is incorrect."
)

st.subheader("Algorithms")
st.markdown(
    """
    - **Isolation Forest:** isolates records using random feature partitions.
    - **LOF:** compares local density with neighbouring records.
    - **One-Class SVM:** learns a normal-data boundary.
    - **Autoencoder:** measures reconstruction error.
    - **KNN distance:** measures distance to a kth neighbour.
    - **DBSCAN:** treats low-density noise points as anomalies.
    """
)

st.subheader("Dataset and validation")
st.write(
    "AHS-CAB 2014 is a proposed primary Indian dataset candidate only if "
    "suitable individual-level files are available. Leave-one-state-out "
    "validation requires a verified state identifier; random splitting must "
    "not be presented as state-held-out validation."
)

st.subheader("Limitations")
st.markdown(
    """
    - Ground-truth anomaly labels are often unavailable.
    - Results depend on preprocessing and parameter choices.
    - No universal best algorithm is claimed.
    - District aggregates must not be treated as individual patient records.
    - Statistical anomalies require qualified human review.
    """
)

st.subheader("Leave-one-state-out validation")
if "raw_dataset" in st.session_state:
    raw = st.session_state["raw_dataset"]
    options = ["None"] + list(raw.columns)
    state_column = st.selectbox("State identifier column", options)
    if state_column == "None":
        result = check_leave_state_out(raw)
    else:
        result = check_leave_state_out(raw, state_column)
    if result["available"]:
        st.success(
            f"State-held-out validation is structurally possible across "
            f"{result['state_count']} state groups."
        )
        st.write("Identified state values: " + ", ".join(map(str, result["states"])))
    else:
        st.warning(result["reason"])
