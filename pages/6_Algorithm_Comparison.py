import pandas as pd
import plotly.express as px
import streamlit as st

from src.evaluation import calculate_metrics


st.set_page_config(page_title="Algorithm Comparison", page_icon="C", layout="wide")
st.title("Algorithm Comparison")

if "model_results" not in st.session_state or "processed_dataset" not in st.session_state:
    st.info("Run anomaly detection before comparing algorithms.")
    st.stop()

processed = st.session_state["processed_dataset"]
raw = st.session_state["raw_dataset"]
results = st.session_state["model_results"]

label_column = st.selectbox(
    "Optional verified ground-truth column",
    ["None"] + list(raw.columns),
)

summary = []
metric_tables = {}
for name, result in results.items():
    predictions = result["predictions"]
    row = {
        "Algorithm": name,
        "Alert rate": round(predictions.mean(), 4),
        "Flagged": int(predictions.sum()),
        "Processing seconds": round(result["processing_time"], 4),
    }
    if label_column != "None":
        labels = pd.to_numeric(raw[label_column], errors="coerce")
        if labels.isna().all():
            st.warning(f"The selected label column '{label_column}' is not numeric.")
        else:
            valid = labels.isin([0, 1]) & labels.notna()
            y_true = labels.loc[valid].astype(int).to_numpy()
            positions = processed.index.get_indexer(valid.index)
            positions = positions[positions >= 0]
            y_pred = predictions[positions]
            y_score = result["scores"][positions]
            if len(y_true) == len(y_pred) and len(set(y_true)) == 2:
                metrics = calculate_metrics(y_true, y_pred, y_score)
                if result.get("supports_score_auc") is False:
                    metrics["pr_auc"] = None
                    metrics["roc_auc"] = None
                row.update(metrics)
                metric_tables[name] = metrics
    summary.append(row)

st.subheader("Comparison summary")
st.dataframe(pd.DataFrame(summary), use_container_width=True)

if label_column == "None":
    st.info(
        "Verified labels are not available. Precision, recall, F1, PR-AUC and "
        "ROC-AUC are not invented; alert rate and algorithm agreement are reported."
    )

fig = px.bar(
    pd.DataFrame(summary),
    x="Algorithm",
    y="Flagged",
    title="Number of records flagged",
)
st.plotly_chart(fig, use_container_width=True)

agreement = pd.DataFrame(
    {name: result["predictions"] for name, result in results.items()}
)
agreement_counts = agreement.sum(axis=1).value_counts().sort_index()
st.subheader("Algorithm agreement")
st.plotly_chart(
    px.bar(
        x=agreement_counts.index.astype(str),
        y=agreement_counts.values,
        labels={"x": "Algorithms flagging the record", "y": "Records"},
    ),
    use_container_width=True,
)

for name, metrics in metric_tables.items():
    st.subheader(f"{name} verified-label metrics")
    st.json({key: value for key, value in metrics.items() if key != "confusion_matrix"})
    st.write("Confusion matrix: [[TN, FP], [FN, TP]]")
    st.write(metrics["confusion_matrix"])

