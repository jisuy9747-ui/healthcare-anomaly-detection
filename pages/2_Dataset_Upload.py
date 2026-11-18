import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import dataset_summary, read_uploaded_dataset


st.set_page_config(page_title="Dataset Upload", page_icon="D", layout="wide")
st.title("Dataset Upload")

with open("data/sample/synthetic_healthcare_demo.csv", "rb") as demo_file:
    st.download_button(
        "Download clearly labeled synthetic demonstration CSV",
        demo_file.read(),
        file_name="synthetic_healthcare_demo.csv",
        mime="text/csv",
        help="This is fabricated demonstration data, not real patient or survey data.",
    )

uploaded = st.file_uploader("Upload CSV or Excel data", type=["csv", "xlsx", "xls"])

if uploaded is not None:
    try:
        df = read_uploaded_dataset(uploaded)
        if df.empty:
            st.error("The uploaded file contains no records.")
        elif len(df.columns) == 0:
            st.error("The uploaded file contains no columns.")
        else:
            st.session_state["raw_dataset"] = df.copy(deep=True)
            st.success(f"Loaded {len(df):,} records and {len(df.columns):,} columns.")
    except Exception as exc:
        st.error(str(exc))

if "raw_dataset" in st.session_state:
    df = st.session_state["raw_dataset"]
    summary = dataset_summary(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{summary['rows']:,}")
    c2.metric("Columns", f"{summary['columns']:,}")
    c3.metric("Duplicate rows", f"{summary['duplicates']:,}")
    c4.metric("Numerical columns", len(summary["numeric_columns"]))

    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Columns", "Missing values", "Statistics"])
    with tab1:
        column_info = pd.DataFrame(
            {
                "Column": df.columns,
                "Data type": [str(dtype) for dtype in df.dtypes],
                "Kind": [
                    "Numerical" if column in summary["numeric_columns"] else "Categorical"
                    for column in df.columns
                ],
            }
        )
        st.dataframe(column_info, use_container_width=True)
        if summary["constant_columns"]:
            st.warning(
                "Constant columns detected: " + ", ".join(summary["constant_columns"])
            )

    with tab2:
        missing_df = pd.DataFrame(
            {
                "Missing count": summary["missing_counts"],
                "Missing percent": summary["missing_percentage"],
            }
        ).fillna(0)
        st.dataframe(missing_df, use_container_width=True)
        if not summary["missing_counts"].empty:
            fig = px.bar(
                missing_df.reset_index(names="Column"),
                x="Missing count",
                y="Column",
                orientation="h",
                title="Missing values by column",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing values were detected.")

    with tab3:
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    st.warning(
        "This view is a general record-level upload. District-level aggregates "
        "or mixed survey sources must not be treated as individual patient records."
    )

