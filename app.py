import streamlit as st
import pandas as pd
import pickle
import json
import os
import re

st.set_page_config(page_title="ChatGPT Review Topic Intelligence", layout="wide")
st.title("ChatGPT Review Topic Intelligence System")

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

# Helper to summarize topics and get sample reviews
def summarize_topics(df, lda_labels, nmf_labels):
    summary = "SUMMARY OF TOPIC MODELING RESULTS FOR USER REVIEWS\n\n"
    
    # LDA Topics Summary
    summary += "--- LDA Model Discovered Topics ---\n"
    lda_counts = df["LDA_Topic"].value_counts()
    for topic, count in lda_counts.items():
        summary += f"Topic Label: {topic} | Count of Reviews: {count}\n"
        # Get top 3 actual review contents for this topic
        samples = df[df["LDA_Topic"] == topic]["content"].dropna().head(3).tolist()
        summary += "Representative Reviews:\n"
        for i, sample in enumerate(samples):
            clean_sample = str(sample).replace("\n", " ").strip()[:200]
            summary += f"  - Review {i+1}: \"{clean_sample}\"\n"
        summary += "\n"
        
    # NMF Topics Summary
    summary += "--- NMF Model Discovered Topics ---\n"
    nmf_counts = df["NMF_Topic"].value_counts()
    for topic, count in nmf_counts.items():
        summary += f"Topic Label: {topic} | Count of Reviews: {count}\n"
        # Get top 3 actual review contents for this topic
        samples = df[df["NMF_Topic"] == topic]["content"].dropna().head(3).tolist()
        summary += "Representative Reviews:\n"
        for i, sample in enumerate(samples):
            clean_sample = str(sample).replace("\n", " ").strip()[:200]
            summary += f"  - Review {i+1}: \"{clean_sample}\"\n"
        summary += "\n"
        
    return summary

# Helper to call LLM APIs
def generate_prd_via_api(provider, api_key, topic_summary_text):
    prompt = f"""
You are a Senior Product Manager. Your task is to analyze the provided summary of topic modeling results (from LDA and NMF models running on a large corpus of user reviews) and generate a comprehensive, highly actionable Product Requirements Document (PRD).

The topic modeling summary is as follows:
---
{topic_summary_text}
---

Your PRD must focus on solving the user pain points and issues identified in the reviews. It must be formatted in beautiful Markdown and contain the following sections:

1. **Title**: A professional title for the PRD based on the themes discovered.
2. **Executive Summary**: A brief (2-3 sentences) description of what the PRD addresses based on the user feedback.
3. **Feature Specifications & Functional Requirements**:
   Present a list of the key features/solutions needed to address these user pain points.
   For **EACH FEATURE**, you MUST provide a detailed breakdown using the following sub-headings:
   - **Feature Name & Description**: What is the feature and how does it solve a specific topic/pain point from the feedback?
   - **Clear Requirements**: Specific, detailed, and unambiguous technical/functional specifications for implementing the feature.
   - **Prioritization**: High, Medium, or Low prioritization, with a 1-sentence rationale based on the review frequencies.
   - **Tradeoffs**: Technical or product tradeoffs considered for this feature (e.g. build vs buy, performance vs complexity, security vs convenience).
   - **Acceptance Criteria**: Testable and measurable criteria to verify the feature works as expected.
   - **Confidence Score**: Present a feature-specific confidence score (as a percentage, e.g. "85%") estimating how strongly supported this feature is by the review data and how likely it is to succeed. Add a 1-sentence explanation of this score.

Ensure the document is structured professionally, contains no generic placeholders, and uses technical terminology where appropriate.
"""

    if provider == "OpenAI":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a senior product manager drafting a detailed technical PRD from customer feedback analytics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ **OpenAI API Error**: {e}"
            
    elif provider == "Google Gemini":
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ **Google Gemini API Error**: {e}"
            
    return "Unsupported provider."

DEFAULT_MOCK_PRD = """# Product Requirements Document (PRD)
## ChatGPT Mobile Client Optimization & Feature Enhancements

### Executive Summary
This PRD outlines critical product changes to address core user complaints discovered in ChatGPT app reviews, specifically targeting response accuracy, billing transparency, mobile performance stability, and content generation limitations.

---

### Feature 1: Mobile Client Reliability & Performance Patch
- **Feature Name & Description**: Stability and local cache optimization for the iOS/Android applications. Directly addresses the "Technical Performance", "Mobile App Issues", and "Technical Bugs" topics.
- **Clear Requirements**:
  - Implement dynamic fallback queries when network timeout exceeds 4.5 seconds.
  - Compress offline database state using Room/SQLite vacuuming to prevent memory bloat on low-end mobile devices.
  - Implement progressive loading for long chat histories rather than loading the full thread on startup.
- **Prioritization**: **High** (Essential to prevent app churn and negative store reviews due to crashes).
- **Tradeoffs**: Decreases local chat search speed slightly in exchange for a 40% reduction in app startup crash rate.
- **Acceptance Criteria**:
  - App startup time must be under 1.8 seconds on mid-range devices.
  - Network timeout failures must gracefully display a retry button rather than freezing or crashing the interface.
- **Confidence Score**: **92%** (Highly supported by "Mobile App Issues" which represents 18% of the critical review corpus).

---

### Feature 2: AI Code Generation & Factual Verification Guardrails
- **Feature Name & Description**: Interactive code execution and citations feature to solve the "AI Response Quality" and "Answer Quality & Accuracy" complaints.
- **Clear Requirements**:
  - Add an "Execute Code" sandboxed environment button inside code blocks.
  - Integrate a real-time web verification widget showing trust sources when answering fact-sensitive questions.
- **Prioritization**: **High** (Directly addresses ChatGPT's main utility value - response quality and reliability).
- **Tradeoffs**: Requires hosting isolated container runtimes for code execution, increasing infrastructure costs.
- **Acceptance Criteria**:
  - Code execution must complete in under 2.5 seconds.
  - Web verification tooltips must link directly to the fetched source pages.
- **Confidence Score**: **89%** (Backed by "Accuracy & Reliability" and "Content Generation" reviews complaining about hallucinations).

---

### Feature 3: Smart Billing Alerts & Subscription Management Portal
- **Feature Name & Description**: Self-service billing management and proactive renewal alerts to resolve the "Subscription & Pricing" and "Payment & Subscription" topics.
- **Clear Requirements**:
  - Send email/push notifications 3 days prior to subscription auto-renew.
  - Build an in-app payment dashboard allowing instant subscription pauses and credit card updates.
- **Prioritization**: **Medium** (Reduces billing-related customer support tickets, which represents 12% of the critical reviews).
- **Tradeoffs**: Giving users an easy way to pause subscriptions might slightly increase churn rate in the short term, but improves long-term brand trust.
- **Acceptance Criteria**:
  - In-app cancellation flows must be completed in no more than 3 steps.
  - Billing status changes must reflect in user accounts within 200ms of transaction confirmation.
- **Confidence Score**: **95%** (Backed by the highest proportion of negative reviews complaining about accidental double-billing).

---

### Feature 4: Mobile Interface Usability and Customization
- **Feature Name & Description**: Customizable font sizes, high-contrast dark modes, and voice input sensitivity settings to improve the "User Experience" and "Interface & Usability" topics.
- **Clear Requirements**:
  - Add an accessibility settings panel in the app menu.
  - Support system-wide dynamic text size scaling.
  - Implement a noise-gate slider for the voice dictation button.
- **Prioritization**: **Low** (Improves usability but is secondary to core reliability and intelligence issues).
- **Tradeoffs**: Increases UI codebase complexity and QA testing load across different device screen sizes.
- **Acceptance Criteria**:
  - The UI must render correctly at up to 200% font scaling without text overlap or cutoff.
- **Confidence Score**: **78%** (Supported by accessibility requests in the "User Experience" topic).
"""

# Sidebar Configuration
st.sidebar.title("🛠️ Configuration")

# API Settings Expander
with st.sidebar.expander("🔑 LLM API Configuration", expanded=True):
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["OpenAI", "Google Gemini"],
        help="Select the AI service to generate the PRD."
    )
    
    api_key_placeholder = ""
    if llm_provider == "OpenAI":
        api_key_placeholder = os.environ.get("OPENAI_API_KEY", "")
    else:
        api_key_placeholder = os.environ.get("GEMINI_API_KEY", "")
        
    api_key = st.text_input(
        f"{llm_provider} API Key",
        value=api_key_placeholder,
        type="password",
        help=f"Enter your {llm_provider} API Key. Leave blank to run in Demo Mode using the default ChatGPT reviews."
    )

st.write("Upload a CSV of ChatGPT reviews and automatically detect topics using LDA and NMF models.")

uploaded_file = st.file_uploader("Upload CSV containing a 'content' column", type=["csv"])

# Trigger model prediction when file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if "content" not in df.columns:
        st.error("CSV must contain a column named 'content'.")
    else:
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        
        # We want to run Topic Modeling automatically or on click
        # Since the original app ran on button click, we keep the button but store results in session state
        # so they can persist for PRD generation.
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
            
            # Save results in session state
            st.session_state["topic_modeling_done"] = True
            st.session_state["topic_df"] = df
            st.session_state["topic_summary_text"] = summarize_topics(df, lda_labels, nmf_labels)
            
            st.success("Topic modeling complete!")
            
        # Display results if completed
        if st.session_state.get("topic_modeling_done", False):
            topic_df = st.session_state["topic_df"]
            
            st.subheader("Topic Modeling Results")
            
            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**LDA Topic Distribution**")
                lda_counts = topic_df["LDA_Topic"].value_counts().reset_index()
                lda_counts.columns = ["Topic", "Count"]
                lda_counts = lda_counts.sort_values("Count", ascending=False)
                st.dataframe(lda_counts, use_container_width=True)
                
            with col2:
                st.write("**NMF Topic Distribution**")
                nmf_counts = topic_df["NMF_Topic"].value_counts().reset_index()
                nmf_counts.columns = ["Topic", "Count"]
                nmf_counts = nmf_counts.sort_values("Count", ascending=False)
                st.dataframe(nmf_counts, use_container_width=True)
                
            # Download option
            with st.expander("Download Full Results (includes content + per-row topics)"):
                csv = topic_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Full Results as CSV",
                    csv,
                    "topic_model_output.csv",
                    "text/csv"
                )
                
            # --- PRD Generation Section ---
            st.markdown("---")
            st.subheader("🚀 AI PRD Generator from Topics")
            st.write(
                "Transform the discovered topic modeling insights (distributions and sample reviews) "
                "into a structured Product Requirements Document (PRD) containing feature-specific "
                "requirements, tradeoffs, acceptance criteria, and confidence scores."
            )
            
            if st.button("🚀 Generate PRD from Discovered Pain-Points"):
                summary_text = st.session_state["topic_summary_text"]
                prd_text = ""
                
                # Check if API Key is empty
                if not api_key:
                    st.info("ℹ️ Running in **Demo Mode** using the pre-compiled PRD for ChatGPT app reviews.")
                    prd_text = DEFAULT_MOCK_PRD
                else:
                    with st.spinner(f"Generating PRD using {llm_provider}..."):
                        prd_text = generate_prd_via_api(llm_provider, api_key, summary_text)
                        
                if prd_text:
                    if prd_text.startswith("❌"):
                        st.error(prd_text)
                    else:
                        st.success("🎉 PRD Generated Successfully!")
                        st.markdown(prd_text)
                        
                        st.download_button(
                            label="📥 Download PRD as Markdown",
                            data=prd_text,
                            file_name="PRD_from_topics.md",
                            mime="text/markdown"
                        )
else:
    # Clear session state if file is removed
    if "topic_modeling_done" in st.session_state:
        del st.session_state["topic_modeling_done"]
