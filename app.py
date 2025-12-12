import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="ChatGPT Review Topic Intelligence", layout="wide")
st.title("ChatGPT Review Topic Intelligence System")

st.write("Upload a CSV of ChatGPT reviews and automatically detect topics using LDA and NMF models.")

# Load models
with open("models/lda_model.pkl", "rb") as f:
    lda = pickle.load(f)
with open("models/nmf_model.pkl", "rb") as f:
    nmf = pickle.load(f)

with open("models/count_vectorizer.pkl", "rb") as f:
    count_vec = pickle.load(f)
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vec = pickle.load(f)

with open("models/lda_topic_labels.pkl", "rb") as f:
    lda_labels = pickle.load(f)
with open("models/nmf_topic_labels.pkl", "rb") as f:
    nmf_labels = pickle.load(f)

uploaded_file = st.file_uploader("Upload CSV containing a 'content' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "content" not in df.columns:
        st.error("CSV must contain a column named 'content'.")
    else:
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Run Topic Modeling"):
            st.subheader("Processing...")

            # LDA Predictions
            bow = count_vec.transform(df["content"].astype(str))
            lda_topics = lda.transform(bow).argmax(axis=1)
            df["LDA_Topic"] = [lda_labels[f"Topic {i+1}"] for i in lda_topics]

            # NMF Predictions
            tfidf = tfidf_vec.transform(df["content"].astype(str))
            nmf_topics = nmf.transform(tfidf).argmax(axis=1)
            df["NMF_Topic"] = [nmf_labels[f"Topic {i+1}"] for i in nmf_topics]

            st.success("Topic modeling complete!")

            st.subheader("Topic Modeling Results")

            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)

            with col1:
                st.write("**LDA Topic Distribution**")
                lda_counts = df["LDA_Topic"].value_counts().reset_index()
                lda_counts.columns = ["Topic", "Count"]
                lda_counts = lda_counts.sort_values("Count", ascending=False)
                st.dataframe(lda_counts, use_container_width=True)

            with col2:
                st.write("**NMF Topic Distribution**")
                nmf_counts = df["NMF_Topic"].value_counts().reset_index()
                nmf_counts.columns = ["Topic", "Count"]
                nmf_counts = nmf_counts.sort_values("Count", ascending=False)
                st.dataframe(nmf_counts, use_container_width=True)

            # Download option (hidden by default)
            with st.expander("Download Full Results (includes content + per-row topics)"):
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Full Results as CSV",
                    csv,
                    "topic_model_output.csv",
                    "text/csv"
                )
