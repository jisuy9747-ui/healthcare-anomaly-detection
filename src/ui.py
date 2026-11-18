import streamlit as st


def apply_theme():
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 92% 5%, rgba(61, 164, 145, 0.11), transparent 22rem),
                linear-gradient(145deg, #f8fbfa 0%, #eef5f3 100%);
        }
        [data-testid="stSidebar"] {
            background: #0d2f2a;
            border-right: 1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] * {
            color: #effaf7;
        }
        [data-testid="stSidebar"] [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stHeader"] {
            background: rgba(248, 251, 250, 0.82);
            backdrop-filter: blur(12px);
        }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,.96);
            border: 1px solid #dfeae7;
            border-top: 4px solid #2f806f;
            border-radius: 16px;
            padding: 20px 18px 16px;
            box-shadow: 0 12px 32px rgba(26, 69, 61, .07);
        }
        [data-testid="stMetricLabel"] {
            color: #60736e;
            font-weight: 700;
            letter-spacing: .02em;
        }
        [data-testid="stMetricValue"] {
            color: #123c34;
            font-size: 30px;
        }
        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid #dce8e5;
            border-radius: 22px;
            padding: 34px 38px;
            margin: 4px 0 24px;
            background:
                linear-gradient(115deg, rgba(255,255,255,.97), rgba(239,248,245,.94)),
                radial-gradient(circle at right, #56b5a0, transparent);
            box-shadow: 0 18px 44px rgba(25, 76, 66, .08);
        }
        .hero h1 {
            color: #123c34;
            font-size: clamp(32px, 4vw, 52px);
            line-height: 1.08;
            margin: 0 0 14px;
        }
        .hero p {
            max-width: 760px;
            color: #526964;
            font-size: 17px;
            margin: 0;
        }
        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 12px;
            margin-bottom: 18px;
            border-radius: 999px;
            background: #e7f4ef;
            color: #23695a;
            font-size: 13px;
            font-weight: 800;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #39a485;
            box-shadow: 0 0 0 4px rgba(57,164,133,.12);
        }
        .workflow-card {
            min-height: 150px;
            background: white;
            border: 1px solid #dfeae7;
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 10px 28px rgba(25, 76, 66, .06);
        }
        .workflow-card .step {
            width: 34px;
            height: 34px;
            display: grid;
            place-items: center;
            border-radius: 10px;
            background: #dff1eb;
            color: #246a5a;
            font-weight: 900;
            margin-bottom: 12px;
        }
        .workflow-card h3 {
            color: #173f37;
            font-size: 17px;
            margin: 0 0 7px;
        }
        .workflow-card p {
            color: #667873;
            font-size: 14px;
            margin: 0;
        }
        .notice-card {
            border-left: 5px solid #3a9a82;
            border-radius: 12px;
            background: #edf7f3;
            color: #385b52;
            padding: 15px 18px;
            margin: 8px 0 18px;
        }
        .section-heading {
            color: #173f37;
            font-size: 21px;
            font-weight: 800;
            margin: 28px 0 12px;
        }
        .stButton > button,
        .stLinkButton > a {
            border-radius: 11px;
            border: 1px solid #cfe0db;
            background: white;
            color: #1d5d50;
            font-weight: 750;
            min-height: 44px;
            box-shadow: 0 5px 16px rgba(25,76,66,.05);
        }
        .stButton > button:hover,
        .stLinkButton > a:hover {
            border-color: #65a897;
            color: #123c34;
        }
        [data-testid="stExpander"] {
            border: 1px solid #dde8e5;
            border-radius: 12px;
            background: rgba(255,255,255,.88);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

