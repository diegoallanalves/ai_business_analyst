from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
KNOWLEDGE = DATA / "knowledge"

def load_portfolio():
    return pd.read_csv(DATA / "portfolio.csv")

def calculate_opportunity(df):
    x = df.copy()
    complaint_score = (x["complaints"].clip(upper=150) / 150) * 100
    x["opportunity_score"] = (
        x["cost_growth_pct"] * 1.4
        + x["manual_work_pct"] * 0.45
        + complaint_score * 0.18
        + x["revenue_growth_pct"] * 0.35
        + x["data_quality"] * 0.12
    ).clip(0, 100).round(1)
    return x

def company_metrics(company):
    df = calculate_opportunity(load_portfolio())
    row = df[df.company.str.lower() == company.lower()]
    if row.empty:
        raise ValueError(f"Unknown company: {company}")
    return row.iloc[0].to_dict()

def compare_portfolio():
    return calculate_opportunity(load_portfolio()).sort_values(
        "opportunity_score", ascending=False
    ).to_dict("records")

def retrieve_context(company, question, top_k=2):
    """Small local RAG (Retrieval-Augmented Generation) retriever using TF-IDF."""
    docs = []
    for path in KNOWLEDGE.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        if company.lower() in text.lower() or company.lower() in path.stem.lower():
            docs.append((path.name, text))
    if not docs:
        return []

    chunks = []
    for name, text in docs:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
        chunks.extend([(name, p) for p in parts])

    corpus = [question] + [c[1] for c in chunks]
    vectors = TfidfVectorizer(stop_words="english").fit_transform(corpus)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    ranked = scores.argsort()[::-1][:top_k]
    return [{"source": chunks[i][0], "text": chunks[i][1], "score": round(float(scores[i]), 3)}
            for i in ranked]

def root_cause_assessment(metrics):
    reasons = []
    if metrics["complaints"] >= 80:
        reasons.append("Customer complaints are high.")
    if metrics["manual_work_pct"] >= 55:
        reasons.append("A large share of operations remains manual.")
    if metrics["cost_growth_pct"] > metrics["revenue_growth_pct"]:
        reasons.append("Operating costs are growing faster than revenue.")
    if metrics["data_quality"] < 80:
        reasons.append("Data quality should improve before advanced AI is scaled.")
    return reasons

def ai_suitability(metrics):
    evidence = root_cause_assessment(metrics)
    if metrics["manual_work_pct"] >= 55 and metrics["data_quality"] >= 70:
        status = "Potential fit — validate root cause first"
    elif metrics["data_quality"] < 70:
        status = "Not ready — improve data foundations first"
    else:
        status = "Unclear — investigate before selecting AI"
    return {"status": status, "evidence": evidence}
