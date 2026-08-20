import os
import json
from dotenv import load_dotenv
from .tools import compare_portfolio, company_metrics, retrieve_context, ai_suitability

load_dotenv()

def classify_intent(question):
    q = question.lower()
    if any(x in q for x in ["right solution", "right technology", "should we use ai", "is ai", "need ai", "appropriate", "suitable"]):
        return "ai_suitability"
    if any(x in q for x in ["which company", "prioritize", "portfolio", "strongest opportunity"]):
        return "portfolio_comparison"
    if any(x in q for x in ["why", "root cause", "complaint", "cost", "margin"]):
        return "root_cause"
    if any(x in q for x in ["roi", "return", "saving", "payback", "value"]):
        return "business_value"
    return "general_investigation"

def choose_company(question, selected_company):
    """Resolve company safely: an explicit company in the question beats the dropdown."""
    q = question.lower()
    portfolio = compare_portfolio()
    for row in portfolio:
        if row["company"].lower() in q:
            return row["company"], "question"
    if classify_intent(question) == "portfolio_comparison":
        return portfolio[0]["company"], "portfolio_ranking"
    return selected_company, "dropdown"

def deterministic_answer(question, company, intent, metrics, rag, suitability):
    evidence = suitability["evidence"]
    evidence_text = " ".join(evidence) if evidence else "The current metrics do not establish a clear root cause."
    if intent == "ai_suitability":
        return {"decision": f"Do not assume AI is the answer for {company}. Validate the root cause first.", "why": f"{company} has {int(metrics['complaints'])} complaints, but complaint volume alone does not prove that AI is the correct technology. {evidence_text}", "next_investigation": "Segment the complaints by cause. Measure how many come from repetitive manual exception handling, data-quality problems, broken integrations, product defects or policy/process issues.", "ai_pilot": "Only if repetitive manual investigation is a major root cause: test AI-assisted exception triage on a small sample with human approval.", "kpi": "Resolution time, manual handling time, complaint rate, error rate and escalation rate.", "risk_control": "Human approval for high-impact decisions; audit logs; access controls; no autonomous payment decision.", "suggested_approach": "Baseline → root-cause analysis → small controlled pilot → measure → scale only if value is proven."}
    if intent == "root_cause":
        return {"decision": f"Investigate the operating drivers behind {company}'s performance before prescribing technology.", "why": evidence_text, "next_investigation": "Break the process into steps and identify where time, rework, errors and complaints are created.", "ai_pilot": "Select AI only for a validated repetitive, information-heavy task.", "kpi": "Cycle time, cost per case, rework, complaints and exception rate.", "risk_control": "Keep calculations deterministic and require evidence for recommendations.", "suggested_approach": "Measure → diagnose → redesign → automate/AI where justified → monitor."}
    if intent == "business_value":
        return {"decision": f"Build a quantified business case for {company} before scaling.", "why": evidence_text, "next_investigation": "Collect case volume, handling minutes, loaded staff cost, error cost and current service level.", "ai_pilot": "Run a controlled pilot and compare against the baseline.", "kpi": "Hours saved, cost avoided, resolution time, quality, adoption and payback period.", "risk_control": "Do not count estimated savings as realized value until validated.", "suggested_approach": "Baseline → pilot cost → measured benefit → ROI (Return on Investment) → scale decision."}
    return {"decision": f"Prioritize {company} for the next value-creation assessment.", "why": f"Opportunity score: {metrics['opportunity_score']}/100. {evidence_text}", "next_investigation": "Validate the problem with users and process data.", "ai_pilot": "AI-assisted workflow triage with human approval.", "kpi": "Manual handling time, resolution time and complaint volume.", "risk_control": "Audit logs, access controls and human approval for high-impact actions.", "suggested_approach": "Start small, establish a baseline, measure the KPI and scale only if measurable value is demonstrated."}

def llm_answer(question, company, intent, metrics, rag, suitability, fallback):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return fallback, False
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        payload = {"question": question, "company": company, "intent": intent, "trusted_metrics": metrics, "retrieved_context": rag, "ai_suitability": suitability}
        system = """You are an AI value-creation analyst for financial-services portfolio companies.
Answer the user's actual question. Never assume AI is the right solution.
Use only the trusted metrics and retrieved context supplied.
Distinguish evidence from hypothesis. Recommend root-cause analysis when evidence is insufficient.
Critical calculations and high-impact financial decisions must remain deterministic/human-controlled.
Return strict JSON with keys: decision, why, next_investigation, ai_pilot, kpi, risk_control, suggested_approach.
Keep each value concise."""
        response = client.chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0.2, messages=[{"role":"system","content":system},{"role":"user","content":json.dumps(payload)}], response_format={"type":"json_object"})
        return json.loads(response.choices[0].message.content), True
    except Exception:
        return fallback, False

def investigate(question, selected_company):
    intent = classify_intent(question)
    company, company_source = choose_company(question, selected_company)
    metrics = company_metrics(company)
    rag = retrieve_context(company, question)
    suitability = ai_suitability(metrics)
    tools = []
    if intent == "portfolio_comparison":
        tools += ["Portfolio Comparison Tool", "Company Selection Tool"]
    tools += ["Financial Analysis Tool", "Question-aware RAG Search Tool"]
    if intent == "ai_suitability":
        tools += ["Root Cause Analysis Tool", "AI Suitability Assessment Tool"]
    elif intent == "business_value":
        tools += ["Business Value Assessment Tool"]
    else:
        tools += ["Recommendation Tool"]
    fallback = deterministic_answer(question, company, intent, metrics, rag, suitability)
    answer, used_llm = llm_answer(question, company, intent, metrics, rag, suitability, fallback)
    return {"intent": intent, "company": company, "company_source": company_source, "metrics": metrics, "rag": rag, "suitability": suitability, "tools": tools, "answer": answer, "used_llm": used_llm}
