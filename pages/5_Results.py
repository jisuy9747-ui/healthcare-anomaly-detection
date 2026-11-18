import pandas as pd
import plotly.express as px
import streamlit as st

from src.export import dataframe_to_csv, dataframe_to_excel


st.set_page_config(page_title="Results", page_icon="R", layout="wide")
st.title("Results")

if "model_results" not in st.session_state or "processed_dataset" not in st.session_state:
    st.info("Run anomaly detection before viewing integrated results.")
    st.stop()

processed = st.session_state["processed_dataset"]
raw = st.session_state["raw_dataset"]
results = st.session_state["model_results"]

algorithm = st.selectbox("Display algorithm", list(results.keys()))
result = results[algorithm]
predictions = result["predictions"]

c1, c2, c3 = st.columns(3)
c1.metric("Normal records", int((~predictions).sum()))
c2.metric("Flagged records", int(predictions.sum()))
c3.metric("Anomaly percentage", f"{predictions.mean() * 100:.2f}%")

tab_flagged, tab_normal, tab_distribution = st.tabs(
    ["Flagged records", "Normal records", "Score distribution"]
)

flagged = raw.loc[processed.index[predictions]].copy()
normal = raw.loc[processed.index[~predictions]].copy()

with tab_flagged:
    if len(flagged):
        flagged.insert(0, "Anomaly score", result["scores"][predictions])
        st.dataframe(flagged, use_container_width=True)
        st.download_button(
            "Download flagged records (CSV)",
            flagged.to_csv(index=True).encode("utf-8"),
            file_name=f"{algorithm.lower().replace(' ', '_')}_flagged.csv",
            mime="text/csv",
            key="results_flagged_csv",
        )
    else:
        st.info("No records were flagged by this algorithm.")

with tab_normal:
    st.dataframe(normal, use_container_width=True)

with tab_distribution:
    scored = pd.DataFrame(
        {
            "Anomaly score": result["scores"],
            "Status": ["Flagged" if flag else "Normal" for flag in predictions],
        }
    )
    fig = px.histogram(
        scored,
        x="Anomaly score",
        color="Status",
        nbins=40,
        title=f"{algorithm} anomaly-score distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

selected_features = list(processed.columns)
if len(selected_features) >= 2:
    st.subheader("Two-feature scatter plot")
    x_feature = st.selectbox("Horizontal feature", selected_features, index=0)
    y_feature = st.selectbox(
        "Vertical feature",
        selected_features,
        index=1,
    )
    scatter = processed[[x_feature, y_feature]].copy()
    scatter["Status"] = ["Flagged" if flag else "Normal" for flag in predictions]
    fig = px.scatter(
        scatter,
        x=x_feature,
        y=y_feature,
        color="Status",
        title=f"{algorithm}: {x_feature} versus {y_feature}",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Export all algorithm results")
all_scores = pd.DataFrame({"Record index": processed.index})
for name, model_result in results.items():
    all_scores[f"{name} score"] = model_result["scores"]
    all_scores[f"{name} flagged"] = model_result["predictions"]

st.download_button(
    "Download combined anomaly scores (CSV)",
    all_scores.to_csv(index=False).encode("utf-8"),
    file_name="combined_anomaly_scores.csv",
    mime="text/csv",
)

st.download_button(
    "Download combined anomaly results (Excel)",
    dataframe_to_excel(all_scores),
    file_name="combined_anomaly_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
