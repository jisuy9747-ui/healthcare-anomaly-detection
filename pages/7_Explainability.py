import streamlit as st

from src.explainability import explain_record


st.set_page_config(page_title="Explainability", page_icon="E", layout="wide")
st.title("Explainability")

if "model_results" not in st.session_state or "processed_dataset" not in st.session_state:
    st.info("Run anomaly detection before reviewing flagged records.")
    st.stop()

processed = st.session_state["processed_dataset"]
raw = st.session_state["raw_dataset"]
results = st.session_state["model_results"]

algorithm = st.selectbox("Algorithm", list(results.keys()))
result = results[algorithm]
flagged_positions = [i for i, flag in enumerate(result["predictions"]) if flag]

if not flagged_positions:
    st.info("This algorithm did not flag any records.")
    st.stop()

position = st.selectbox(
    "Flagged record",
    range(len(flagged_positions)),
    format_func=lambda i: f"Record {processed.index[flagged_positions[i]]}",
)
actual_position = flagged_positions[position]
record_index = processed.index[actual_position]
record = raw.loc[record_index]

st.subheader("Flagged record")
st.dataframe(record.to_frame("Value"), use_container_width=True)
st.metric("Anomaly score", f"{result['scores'][actual_position]:.4f}")

review_key = f"review_{algorithm}_{record_index}"
existing = st.session_state.get(review_key, {})
statuses = [
    "Pending review",
    "Confirmed data issue",
    "Valid unusual observation",
    "Needs further investigation",
]
status = st.selectbox(
    "Human review status",
    statuses,
    index=statuses.index(existing.get("status", "Pending review")),
)
comment = st.text_area("Reviewer comment", existing.get("comment", ""))
if st.button("Save review"):
    st.session_state[review_key] = {"status": status, "comment": comment}
    st.success("Review status and comment saved in this session.")

st.subheader("Feature-level statistical context")
explanation = explain_record(record, raw[list(processed.columns)], list(processed.columns))
st.dataframe(explanation, use_container_width=True)
st.caption(
    "These deviations are statistical observations, not causal explanations "
    "or evidence that the record is medically abnormal."
)
