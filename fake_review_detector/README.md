AI-Powered Fake Review Detection And Analytics Platform

    Built an AI-powered review intelligence platform that classifies reviews as genuine or fake using a trained ML model, analyzes sentiment, flags duplicate and spammy content, scores reviewer risk, and generates downloadable PDF/Excel reports through an admin dashboard.

LIVE DEMO
 [Click here to view live dashboard]

Problem Statement
Businesses and marketplaces lose customer trust every day because fake reviews go undetected — misleading buyers, distorting ratings, and letting spam/reviewer fraud slip through unnoticed.
 
Solution
An AI powered platform that detects fake reviews, flags suspicious reviewer behavior, and scores spam and duplicate content — BEFORE it misleads a customer.

 Key Innovations
1.AI-Based Fake Review Classification with Confidence Score
2.Sentiment Analysis (Positive / Negative / Neutral)
3.Duplicate Review Detection via TF-IDF Similarity
4.Spam Keyword & Pattern Detection Engine
5.Fake Reviewer Risk Scoring (Low / Medium / High)
6.PDF, Excel & QR Code Report Generation
7.Admin Dashboard for User & Dataset Monitoring

Tech Stack
Python 3 | Flask | SQLite
Scikit-learn | Pandas | NumPy
Bootstrap 5 | Chart.js
ReportLab | OpenPyXL | qrcode
 
Results
- Fake Review Model Validation Accuracy: ~100% (on bundled sample dataset)
- Model: TF-IDF (1-2 grams) + Logistic Regression
- Duplicate Detection Threshold: 0.85 Cosine Similarity
- 5 Analysis Modules Running in a Single Pipeline

How to Run Locally
1. Clone repo: `git clone YOUR_REPO_URL`
2. Create venv: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install: `pip install -r requirements.txt`
5. Train model: `python ml/train_model.py`
6. Run: `python app.py`
7. Open: `http://127.0.0.1:5000`

**Default admin login:** 'admin' / 'admin123'