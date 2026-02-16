from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from api.rag_engine import MyanmarAgriAssistEngine

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Agriculture_cleaned.csv"
TRAIN_DATA_PATH = BASE_DIR / "data" / "Agriculture.csv"
EDA_DIR = BASE_DIR / "eda"
PIE_PATH = EDA_DIR / "category_pie.png"
BAR_PATH = EDA_DIR / "category_bar.png"
KEYWORD_PATH = EDA_DIR / "category_keywords.json"

# --------------------------------------------------
# Cached loading
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset(rows: int = 5):
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df.head(rows)

@st.cache_data(show_spinner=False)
def load_sample_instruction() -> str:
    source_path = TRAIN_DATA_PATH if TRAIN_DATA_PATH.exists() else DATA_PATH
    if not source_path.exists():
        return ""

    df = pd.read_csv(source_path)
    if "Instruction" not in df.columns:
        return ""

    instructions = df["Instruction"].dropna().astype(str).str.strip()
    instructions = instructions[instructions != ""]
    if instructions.empty:
        return ""

    return instructions.iloc[0]

@st.cache_resource(show_spinner=True)
def load_engine():
    return MyanmarAgriAssistEngine()


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(
    page_title="Myanmar Agri-Assist",
    layout="wide",
    page_icon="🌾"
)

st.title("🌾 Myanmar Agri-Assist")
st.write("AI-powered Burmese agricultural knowledge assistant using **XGBoost classification + FAISS RAG retrieval**.")

engine = load_engine()
sample_instruction = load_sample_instruction()

tabs = st.tabs(["📘 Dataset Overview", "🔍 Topic Classification", "🤖 RAG Assistant"])

# --------------------------------------------------
# TAB 1 — Dataset Overview
# --------------------------------------------------
with tabs[0]:
    st.header("📘 Dataset Overview")

    df_sample = load_dataset()

    if df_sample.empty:
        st.warning("Dataset not found. Please ensure data/Agriculture_cleaned.csv exists.")
    else:
        st.subheader("Sample Rows")
        st.dataframe(df_sample, height=300)

    col1, col2 = st.columns(2)

    with col1:
        if PIE_PATH.exists():
            st.image(str(PIE_PATH), caption="Category Distribution (Pie)")
        else:
            st.info("Run EDA to generate pie chart.")

    with col2:
        if BAR_PATH.exists():
            st.image(str(BAR_PATH), caption="Top Categories (Bar)")
        else:
            st.info("Run EDA to generate bar chart.")

    if KEYWORD_PATH.exists():
        st.subheader("Top Keywords per Category")
        st.json(KEYWORD_PATH.read_text(encoding="utf-8"))
    else:
        st.info("Run EDA keyword extraction first.")

# --------------------------------------------------
# TAB 2 — Topic Classification
# --------------------------------------------------
with tabs[1]:
    st.header("🔍 Topic Classification")
    text = st.text_area(
        "Enter Burmese agricultural instruction or response:",
        value=sample_instruction,
        height=200,
    )

    if st.button("Classify", key="classify_button"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            result = engine.classify_text(text)
            st.success(f"Predicted Category: **{result['category']}**")
            st.metric("Confidence", f"{result['confidence']:.2f}")

# --------------------------------------------------
# TAB 3 — RAG Assistant
# --------------------------------------------------
with tabs[2]:
    st.header("🤖 RAG Question Answering Assistant")

    question = st.text_area(
        "Enter a Burmese agricultural question:",
        value=sample_instruction,
        height=200,
    )
    top_k = st.slider("Number of retrieved references:", 1, 5, 3)

    if st.button("Get Answer", key="ask_button"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            result = engine.ask_question(question, top_k=top_k)

            st.subheader("Generated Answer")
            st.write(result["answer"])

            st.subheader("Top References")

            for i, ctx in enumerate(result["contexts"], start=1):
                st.markdown(
                    f"""
                    **{i}. Category:** {ctx['category']}  
                    **Instruction:** {ctx['instruction']}  
                    **Output:** {ctx['output']}  
                    **Score:** {ctx['score']:.3f}
                    """
                )
