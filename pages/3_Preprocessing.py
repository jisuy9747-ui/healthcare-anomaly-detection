import pandas as pd
import plotly.express as px
import streamlit as st

from src.preprocessing import fit_preprocessing, validate_analysis_structure


st.set_page_config(page_title="Preprocessing", page_icon="P", layout="wide")
st.title("Preprocessing")

if "raw_dataset" not in st.session_state:
    st.info("Upload a dataset on the Dataset Upload page before preprocessing.")
    st.stop()

df = st.session_state["raw_dataset"]
numeric_columns = df.select_dtypes(include="number").columns.tolist()

if not numeric_columns:
    st.error("No numerical columns were detected. Numerical data is required for anomaly detection.")
    st.stop()

with st.sidebar:
    st.subheader("Preprocessing settings")
    selected = st.multiselect(
        "Numerical features",
        numeric_columns,
        default=numeric_columns[: min(8, len(numeric_columns))],
    )
    imputation = st.radio("Missing-value handling", ["Median", "Mean", "Most frequent"])
    scaling = st.radio("Scaling", ["Standard", "Robust"])

st.subheader("Before processing")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Selected features", len(selected))
c3.metric("Missing cells", int(df[selected].isna().sum().sum()) if selected else 0)

warnings = validate_analysis_structure(df, selected)
for warning in warnings:
    st.warning(warning)

if selected:
    constant = [column for column in selected if df[column].nunique(dropna=True) <= 1]
    if constant:
        st.warning("Constant features that will be excluded: " + ", ".join(constant))

if st.button("Apply preprocessing", type="primary", disabled=not selected):
    try:
        result = fit_preprocessing(df, selected, scaling=scaling, imputation=imputation)
        st.session_state["processed_dataset"] = result.data.copy(deep=True)
        st.session_state["preprocessing_result"] = result
        st.success("Preprocessing completed. The original uploaded data was preserved.")
    except Exception as exc:
        st.error(str(exc))

if "processed_dataset" in st.session_state:
    processed = st.session_state["processed_dataset"]
    result = st.session_state["preprocessing_result"]

    st.subheader("After processing")
    c1, c2, c3 = st.columns(3)
    c1.metric("Processed rows", f"{len(processed):,}")
    c2.metric("Processed features", len(processed.columns))
    c3.metric("Missing cells", int(processed.isna().sum().sum()))

    st.dataframe(processed.head(100), use_container_width=True)

    if len(processed.columns) >= 2:
        corr = processed.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Processed feature correlation",
            color_continuous_scale="Teal",
        )
        st.plotly_chart(fig, use_container_width=True)

    csv = processed.to_csv(index=True).encode("utf-8")
    st.download_button(
        "Download processed dataset",
        csv,
        file_name="processed_healthcare_dataset.csv",
        mime="text/csv",
    )

