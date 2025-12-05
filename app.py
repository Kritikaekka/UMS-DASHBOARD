# app.py
import os
import re
import random
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic

import nltk
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------
# SEEDS & NLTK
# -------------------------------------------------
random.seed(42)
np.random.seed(42)

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
lemmatizer = WordNetLemmatizer()

# -------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="UMS Usability Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 UMS Usability Analytics Dashboard")
st.markdown(
    """
This dashboard analyzes **University Management System (UMS)** reviews:

- Runs **Transformer sentiment analysis**  
- Extracts **topics with BERTopic**  
- Maps issues to **Norman’s 10 usability principles**  
- Generates **auto recommendations** for developers  
- Lets you **filter, explore, and export** results
"""
)

# -------------------------------------------------
# CACHED MODELS & UTILITIES
# -------------------------------------------------


@st.cache_resource(show_spinner=True)
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


@st.cache_resource(show_spinner=True)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=True)
def get_norman_principles(embed_model):
    norman_principles = {
        "Visibility of system status": "Keep users informed about what is happening with clear progress and status updates.",
        "Match between system and the real world": "Use familiar language and concepts users recognize from their environment.",
        "User control and freedom": "Allow users to undo/redo and navigate back without losing data.",
        "Consistency and standards": "Design and behavior should be consistent across the portal.",
        "Error prevention": "Design to prevent errors (validation, constraints) before they happen.",
        "Recognition rather than recall": "Show options and information to reduce memory load.",
        "Flexibility and efficiency of use": "Support shortcuts, personalization and efficient workflows.",
        "Aesthetic and minimalist design": "Avoid clutter; present minimal necessary information.",
        "Help users recognize, diagnose, and recover from errors": "Clear actionable error messages and recovery steps.",
        "Help and documentation": "Provide help pages, guides, and tooltips for users.",
    }
    principle_names = list(norman_principles.keys())
    principle_texts = list(norman_principles.values())
    principle_embs = embed_model.encode(
        principle_texts, convert_to_tensor=True, show_progress_bar=False
    )
    return norman_principles, principle_names, principle_embs


# -------------------------------------------------
# PREPROCESSING
# -------------------------------------------------


def clean_for_model(text):
    return str(text).strip()


def clean_for_topic(text):
    t = str(text).lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    tokens = [w for w in t.split() if len(w) > 1]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def estimate_severity(review):
    r = str(review).lower()
    if any(w in r for w in ["crash", "fail", "unable", "not working", "cannot login", "can't login", "lose", "lost"]):
        return "High"
    if any(w in r for w in ["slow", "lag", "delay", "hang", "loading", "takes time"]):
        return "Medium"
    return "Low"


# -------------------------------------------------
# RECOMMENDATIONS
# -------------------------------------------------
recommendation_dict = {
    "Visibility of system status": [
        "Add loading indicators or progress bars for long operations.",
        "Display clear status messages during registration, fee payment, and result loading.",
    ],
    "Match between system and the real world": [
        "Use student-friendly words (e.g., 'Exam Form' instead of internal codes).",
        "Use intuitive icons and labels aligned with university terminology.",
    ],
    "User control and freedom": [
        "Allow users to go back or cancel actions without losing filled data.",
        "Provide undo options for critical operations like course registration.",
    ],
    "Consistency and standards": [
        "Standardize button styles, colors, and layouts across all modules.",
        "Keep terminology consistent across student, faculty, and admin views.",
    ],
    "Error prevention": [
        "Validate inputs in real time (e.g., date formats, roll numbers).",
        "Auto-save forms or warn users before session timeouts.",
    ],
    "Recognition rather than recall": [
        "Provide dropdowns and autosuggestions for course codes and departments.",
        "Show recent or frequently used actions on the dashboard.",
    ],
    "Flexibility and efficiency of use": [
        "Offer quick links for common tasks (attendance, marks, timetable).",
        "Allow keyboard shortcuts or pinned items on the home page.",
    ],
    "Aesthetic and minimalist design": [
        "Remove unused options and reduce information clutter on each page.",
        "Use whitespace and grouping to make content easier to scan.",
    ],
    "Help users recognize, diagnose, and recover from errors": [
        "Show clear error messages with steps to fix the problem.",
        "Provide contact/help links directly from error dialogs.",
    ],
    "Help and documentation": [
        "Add a searchable FAQ or help center with screenshots.",
        "Provide short video or step-by-step guides for key workflows.",
    ],
    "Other/Unmapped": [
        "Manual review required; unclear automatic mapping. Consider reading raw comments."
    ],
}


def get_recs(principle):
    return " | ".join(recommendation_dict.get(principle, ["No recommendation available."]))


# -------------------------------------------------
# CORE ANALYSIS PIPELINE
# -------------------------------------------------


def run_full_analysis(df, review_col="Review", role_col="UserType"):
    sentiment_pipe = load_sentiment_model()
    embed_model = load_embedder()
    norman_principles, principle_names, principle_embs = get_norman_principles(
        embed_model
    )

    # --- Cleaning
    df = df.copy()
    df["clean_for_model"] = df[review_col].apply(clean_for_model)
    df["clean_for_topic"] = df[review_col].apply(clean_for_topic)

    # --- Sentiment
    st.info("Running Transformer sentiment analysis on reviews…")
    texts = df["clean_for_model"].tolist()
    labels = []
    scores = []

    progress = st.progress(0)
    total = len(texts)
    for i, t in enumerate(texts):
        try:
            out = sentiment_pipe(t[:512])[0]
            labels.append(out["label"].lower())
            scores.append(float(out["score"]))
        except Exception:
            labels.append("neutral")
            scores.append(0.0)
        if total > 0:
            progress.progress((i + 1) / total)

    df["hf_sentiment"] = labels
    df["hf_sentiment_score"] = scores

    # --- Negative subset
    negative_df = df[df["hf_sentiment"] == "negative"].copy()

    if len(negative_df) == 0:
        st.warning("No negative reviews detected. Try another file.")
        return df, None, None, None

    # --- BERTopic on negative reviews
    st.info("Fitting BERTopic model on negative reviews…")
    texts_neg = negative_df["clean_for_topic"].astype(str).tolist()
    topic_model = BERTopic(
        embedding_model=embed_model, min_topic_size=10, verbose=False
    )
    topics, probs = topic_model.fit_transform(texts_neg)

    negative_df["bertopic_topic"] = topics
    negative_df["topic_prob"] = [p.max() if p is not None else None for p in probs]

    topic_info = topic_model.get_topic_info()

    # --- Map topics to Norman principles
    topic_map = {}
    topic_sim_scores = {}

    for topic_id in topic_info.Topic:
        if topic_id == -1:
            continue
        top_words = topic_model.get_topic(topic_id)
        if not top_words:
            continue
        rep = " ".join([w for w, _ in top_words[:10]])
        rep_emb = embed_model.encode(rep, convert_to_tensor=True)
        sims = util.cos_sim(rep_emb, principle_embs)[0].cpu().numpy()
        best_idx = sims.argmax()
        topic_map[topic_id] = principle_names[best_idx]
        topic_sim_scores[topic_id] = float(sims[best_idx])

    negative_df["mapped_principle_auto"] = negative_df["bertopic_topic"].map(
        topic_map
    ).fillna("Other/Unmapped")

    # --- Severity & Recommendations
    negative_df["Severity"] = negative_df[review_col].apply(estimate_severity)
    negative_df["Recommendations"] = negative_df["mapped_principle_auto"].apply(
        get_recs
    )

    return df, negative_df, topic_model, topic_info


# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
st.sidebar.header("📂 Upload Reviews File")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel/CSV containing UMS reviews",
    type=["xlsx", "xls", "csv"],
    help="File must contain at least a 'Review' column. Optional: 'UserType' (Student/Faculty).",
)

default_review_col = "Review"
default_role_col = "UserType"

if uploaded_file is not None:
    # Detect file type
    if uploaded_file.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df_raw.head())

    # Column selection
    cols = df_raw.columns.tolist()
    review_col = st.sidebar.selectbox(
        "Select review text column", options=cols, index=cols.index(default_review_col) if default_review_col in cols else 0
    )
    role_col = st.sidebar.selectbox(
        "Select role column (optional)", options=["<None>"] + cols,
        index=(["<None>"] + cols).index(default_role_col) if default_role_col in cols else 0
    )
    role_col = None if role_col == "<None>" else role_col

    if st.sidebar.button("Run Analysis"):
        with st.spinner("Running full usability analysis…"):
            full_df, negative_df, topic_model, topic_info = run_full_analysis(
                df_raw, review_col=review_col, role_col=role_col
            )

        if negative_df is not None:
            st.success("Analysis complete ✅")
            st.markdown("---")

            # -------------------------------------------------
            # METRICS
            # -------------------------------------------------
            col1, col2, col3 = st.columns(3)
            total_reviews = len(full_df)
            neg_reviews = len(negative_df)
            pos_reviews = (full_df["hf_sentiment"] == "positive").sum()

            col1.metric("Total Reviews", total_reviews)
            col2.metric("Negative Reviews", neg_reviews)
            col3.metric("Positive Reviews", pos_reviews)

            # -------------------------------------------------
            # SENTIMENT DISTRIBUTION
            # -------------------------------------------------
            st.subheader("📈 Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x="hf_sentiment", data=full_df, ax=ax)
            ax.set_xlabel("Sentiment (HF Transformer)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # -------------------------------------------------
            # TOPIC OVERVIEW
            # -------------------------------------------------
            st.subheader("🧵 Topic Clusters (Negative Reviews)")
            st.dataframe(topic_info.head(15))

            try:
                st.plotly_chart(topic_model.visualize_topics(), use_container_width=True)
            except Exception:
                st.info("Plotly visual topic map requires plotly; skipping if not installed.")

            # -------------------------------------------------
            # NORMAN PRINCIPLE SUMMARY
            # -------------------------------------------------
            st.subheader("🧠 Norman’s Principles — Issue Counts")

            principle_counts = (
                negative_df["mapped_principle_auto"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "Principle", "mapped_principle_auto": "Count"})
            )

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(
                data=principle_counts, y="Principle", x="Count", ax=ax, palette="magma"
            )
            ax.set_title("Negative Reviews Mapped to Norman’s Principles")
            st.pyplot(fig)

            # Donut chart
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            ax2.pie(
                principle_counts["Count"],
                labels=principle_counts["Principle"],
                autopct="%1.1f%%",
                startangle=140,
                wedgeprops={"width": 0.4},
            )
            ax2.set_title("Overall Usability Issue Distribution")
            st.pyplot(fig2)

            # -------------------------------------------------
            # OPTIONAL ROLE-BASED BREAKDOWN
            # -------------------------------------------------
            if role_col is not None and role_col in negative_df.columns:
                st.subheader("👥 Student vs Faculty Comparison")

                role_summary = (
                    negative_df.groupby(["mapped_principle_auto", role_col])
                    .size()
                    .unstack(fill_value=0)
                )

                st.dataframe(role_summary)

                fig3, ax3 = plt.subplots(figsize=(10, 6))
                role_summary.plot(kind="bar", ax=ax3)
                ax3.set_xlabel("Norman Principle")
                ax3.set_ylabel("Count")
                ax3.set_title("Usability Issues by Principle and Role")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig3)

            # -------------------------------------------------
            # INTERACTIVE FILTERS
            # -------------------------------------------------
            st.subheader("🔍 Explore Individual Negative Reviews")

            # Sidebar filters
            st.sidebar.markdown("---")
            st.sidebar.subheader("Filters")

            unique_principles = sorted(
                negative_df["mapped_principle_auto"].unique().tolist()
            )
            selected_principles = st.sidebar.multiselect(
                "Filter by Norman principle",
                options=unique_principles,
                default=unique_principles,
            )

            severities = sorted(negative_df["Severity"].unique().tolist())
            selected_severities = st.sidebar.multiselect(
                "Filter by severity", options=severities, default=severities
            )

            if role_col is not None and role_col in negative_df.columns:
                roles = sorted(negative_df[role_col].dropna().unique().tolist())
                selected_roles = st.sidebar.multiselect(
                    f"Filter by {role_col}", options=roles, default=roles
                )
            else:
                selected_roles = None

            keyword = st.sidebar.text_input(
                "Keyword search in review text", value=""
            ).strip()

            # Apply filters
            filtered = negative_df.copy()
            filtered = filtered[filtered["mapped_principle_auto"].isin(selected_principles)]
            filtered = filtered[filtered["Severity"].isin(selected_severities)]
            if selected_roles is not None:
                filtered = filtered[filtered[role_col].isin(selected_roles)]
            if keyword:
                filtered = filtered[
                    filtered[review_col].str.contains(keyword, case=False, na=False)
                ]

            st.write(f"Showing **{len(filtered)}** filtered negative reviews:")

            show_cols = [review_col, "hf_sentiment", "hf_sentiment_score",
                         "mapped_principle_auto", "Severity", "Recommendations"]
            if role_col is not None and role_col in filtered.columns:
                show_cols.insert(1, role_col)

            st.dataframe(filtered[show_cols].head(200))

            # -------------------------------------------------
            # EXPORT BUTTONS
            # -------------------------------------------------
            st.subheader("📤 Export Filtered Results")

            # CSV export
            csv_bytes = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download filtered issues as CSV",
                data=csv_bytes,
                file_name="ums_usability_issues_filtered.csv",
                mime="text/csv",
            )

            # Excel export
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                filtered.to_excel(writer, index=False, sheet_name="Issues")
            excel_data = output.getvalue()

            st.download_button(
                label="Download filtered issues as Excel",
                data=excel_data,
                file_name="ums_usability_issues_filtered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

else:
    st.info(
        "👈 Upload an Excel/CSV file with at least a **Review** column to begin."
    )
