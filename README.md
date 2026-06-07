# ChatGPT Review Topic Intelligence System  
### CIS 508 — Machine Learning in Business  
### Final Project — Mokshith Diggenahalli Vasanth Kumar  

---

## 🚀 Project Overview  
This project builds an automated **topic intelligence system** for large-scale ChatGPT app reviews.  
Users upload a CSV file of reviews, and the system applies **LDA** and **NMF** topic models to extract dominant themes and user concerns.

This project demonstrates:  
- End-to-end machine learning workflow  
- Reproducible experiment tracking using **Databricks MLflow**  
- A deployed, interactive **Streamlit web application**  
- Automated topic discovery for real-world decision-making  
- Actionable business insights from user feedback  

---

## 🧠 Machine Learning Models  

### **1. Latent Dirichlet Allocation (LDA)**  
- Built using **CountVectorizer** features  
- 8 interpretable topics  
- Logged metrics: *perplexity*, *log-likelihood*  
- Captures broad themes such as AI response quality, technical performance, pricing, and user experience  

### **2. Non-Negative Matrix Factorization (NMF)**  
- Built using **TF-IDF** features  
- 8 topics with higher semantic coherence  
- Logged metric: *reconstruction error*  
- Captures themes including answer quality, app performance, and content generation  

### **Preprocessing Steps**  
- Text normalization  
- Removal of URLs, punctuation, and noise  
- Duplicate removal  
- Minimum text length filtering  
- Feature extraction with CountVectorizer & TF-IDF  

All models and preprocessing pipelines were trained in **Google Colab** and logged to **Databricks MLflow** for reproducibility.

---

## 📊 Streamlit Web Application  

The deployed Streamlit application allows users to:

- Upload a CSV file containing a `content` column  
- Run automated LDA & NMF topic modeling  
- View **aggregated topic distributions** (professional and safer than row-level predictions)  
- Download a full CSV of predictions  
- **Generate AI PRD Documents**: Feed the topic modeling distributions and representative reviews directly to OpenAI or Google Gemini APIs to generate a structured, feature-by-feature Product Requirements Document (PRD) with prioritization, tradeoffs, acceptance criteria, and confidence scores. Includes a robust "Demo Mode" fallback.

### **Run Locally**
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Repository Structure  

```
chatgpt_topic_app/
│── app.py
│── requirements.txt
│── models/
│   ├── lda_model.pkl
│   ├── nmf_model.pkl
│   ├── count_vectorizer.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── lda_topic_labels.pkl
│   ├── nmf_topic_labels.pkl
│── .gitignore
```

All PKL model files are included for reproducibility and ease of deployment.

---

## 🔗 Project Links  

### **Deployed Streamlit Application**  
https://mokshithdv25-chatgpt-topic-app-app-newq6b.streamlit.app/

### **Databricks MLflow Experiment**  
https://dbc-ea80752b-1ab0.cloud.databricks.com/ml/experiments/36944098176678/runs?o=1853817175437544&searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

### **Google Colab Notebook**  
https://colab.research.google.com/drive/1BbfI7oq42HcpyFprCbfIRfoj2ZsIfC5l?usp=sharing

### **GitHub Repository**  
https://github.com/Mokshithdv25/chatgpt_topic_app

### **Video Presentation**  
https://youtu.be/wEo03d23Oyg

---

### **Slide Deck**  
https://docs.google.com/presentation/d/1xrfJ9JpTBFVdScJOc2JifD4K1AeNbdHgHDwfs7c8NuQ/edit?usp=sharing

---

## 📌 Author  
**Mokshith Diggenahalli Vasanth Kumar**  
Arizona State University  
W. P. Carey School of Business  

