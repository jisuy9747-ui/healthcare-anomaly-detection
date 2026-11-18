import pandas as pd
import plotly.express as px
import streamlit as st

from models.isolation_forest import run_isolation_forest
from models.lof import run_lof
from models.one_class_svm import run_one_class_svm
from models.autoencoder import run_autoencoder
from models.knn import run_knn_distance
from models.dbscan import run_dbscan


st.set_page_config(page_title="Anomaly Detection", page_icon="A", layout="wide")
st.title("Anomaly Detection")

if "processed_dataset" not in st.session_state:
    st.info("Complete preprocessing before running anomaly detection.")
    st.stop()

processed = st.session_state["processed_dataset"]
X = processed.to_numpy()

st.caption(
    "Algorithms score records differently: Isolation Forest and LOF provide "
    "relative unusualness, while One-Class SVM learns a normal-data boundary."
)

selected = st.multiselect(
    "Algorithms",
    [
        "Isolation Forest",
        "Local Outlier Factor",
        "One-Class SVM",
        "Autoencoder",
        "K-Nearest Neighbour",
        "DBSCAN",
    ],
    default=[
        "Isolation Forest",
        "Local Outlier Factor",
        "One-Class SVM",
        "Autoencoder",
        "K-Nearest Neighbour",
        "DBSCAN",
    ],
)

with st.expander("Isolation Forest settings"):
    contamination = st.slider("Contamination", 0.01, 0.30, 0.05, 0.01)

with st.expander("LOF settings"):
    neighbours = st.number_input(
        "Number of neighbours",
        min_value=2,
        max_value=max(2, len(processed) - 1),
        value=min(20, max(2, len(processed) - 1)),
        step=1,
    )
    lof_contamination = st.slider(
        "LOF alert threshold",
        0.01,
        0.30,
        0.05,
0.01,
        help="Used to determine the number of most locally unusual records.",
    )

with st.expander("One-Class SVM settings"):
    kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
    nu = st.slider("nu", 0.01, 0.50, 0.05, 0.01)

with st.expander("Autoencoder settings"):
    epochs = st.slider("Training epochs", 5, 100, 30, 5)
    batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
    threshold_percentile = st.slider("Reconstruction-error percentile", 90.0, 99.5, 95.0, 0.5)

with st.expander("KNN settings"):
    knn_k = st.number_input(
        "Kth neighbour",
        min_value=1,
        max_value=max(1, len(processed) - 1),
        value=min(5, max(1, len(processed) - 1)),
        step=1,
    )

with st.expander("DBSCAN settings"):
    eps = st.slider("eps", 0.1, 5.0, 0.5, 0.1)
    min_samples = st.number_input("Minimum samples", min_value=2, max_value=max(2, len(processed)), value=5, step=1)

if st.button("Run anomaly detection", type="primary", disabled=not selected):
    results = {}
    progress = st.progress(0)
    total = len(selected)
    for index, name in enumerate(selected):
        try:
            if name == "Isolation Forest":
                results[name] = run_isolation_forest(X, contamination=contamination)
            elif name == "Local Outlier Factor":
                results[name] = run_lof(
                    X,
                    neighbours=int(neighbours),
                    contamination=lof_contamination,
                )
            elif name == "One-Class SVM":
                results[name] = run_one_class_svm(X, kernel=kernel, nu=nu)
            elif name == "Autoencoder":
                results[name] = run_autoencoder(
                    X,
                    epochs=epochs,
                    batch_size=batch_size,
                    threshold_percentile=threshold_percentile,
                )
            elif name == "K-Nearest Neighbour":
                results[name] = run_knn_distance(X, k=int(knn_k))
            else:
                results[name] = run_dbscan(X, eps=eps, min_samples=int(min_samples))
        except Exception as exc:
            st.error(f"{name} failed: {exc}")
        progress.progress((index + 1) / total)

    st.session_state["model_results"] = results

if "model_results" in st.session_state:
    results = st.session_state["model_results"]
    raw = st.session_state["raw_dataset"]

    summary_rows = []
    overlap_frames = {}
    for name, result in results.items():
        predictions = result["predictions"]
        alert_count = int(predictions.sum())
        summary_rows.append(
            {
                "Algorithm": name,
                "Flagged": alert_count,
                "Flagged %": round(alert_count / len(processed) * 100, 2),
                "Processing seconds": round(result["processing_time"], 4),
            }
        )
        overlap_frames[name] = pd.Series(predictions, index=processed.index, dtype=bool)

    st.subheader("Anomaly detection results")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    tabs = st.tabs(list(results.keys()))
    for tab, (name, result) in zip(tabs, results.items()):
        with tab:
            predictions = result["predictions"]
            scored = pd.DataFrame(
                {
                    "Record index": processed.index,
                    "Anomaly score": result["scores"],
                    "Flagged": predictions,
                }
            )

            c1, c2 = st.columns(2)
            c1.metric("Flagged records", int(predictions.sum()))
            c2.metric("Flagged percentage", f"{predictions.mean() * 100:.2f}%")

            fig = px.histogram(
                scored,
                x="Anomaly score",
                color="Flagged",
                nbins=40,
                title=f"{name} anomaly-score distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

            flagged_indices = processed.index[predictions]
            if len(flagged_indices):
                flagged = raw.loc[flagged_indices].copy()
                flagged.insert(0, "Anomaly score", result["scores"][predictions])
                st.subheader("Flagged records")
                st.dataframe(flagged, use_container_width=True)
                st.download_button(
                    f"Download {name} flagged records",
                    flagged.to_csv(index=True).encode("utf-8"),
                    file_name=f"{name.lower().replace(' ', '_')}_flagged.csv",
                    mime="text/csv",
                    key=f"download_{name}",
                )
            else:
                st.info("No records were flagged.")

    if len(overlap_frames) > 1:
        agreement = pd.DataFrame(overlap_frames).sum(axis=1)
        st.subheader("Algorithm agreement")
        counts = agreement.value_counts().sort_index()
        fig = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            labels={"x": "Number of algorithms flagging record", "y": "Records"},
            title="Overlap between anomaly-detection algorithms",
        )
        st.plotly_chart(fig, use_container_width=True)

