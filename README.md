# AI Value Creation Agent

Interview-ready Streamlit prototype showing question-aware Agentic AI for financial-services portfolio value creation.

## Key capabilities
- Intent classification and dynamic tool selection.
- Company resolution: a company explicitly named in the question overrides the dropdown.
- Deterministic financial metrics and opportunity scoring.
- Question-aware RAG (Retrieval-Augmented Generation).
- Root-cause / AI-suitability checks before recommending AI.
- Optional OpenAI LLM (Large Language Model) reasoning.
- Interactive portfolio opportunity map and opportunity-share donut chart.

## Install
```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Interview test questions
1. `Suppose AsterPay has 92 complaints. How do you know AI is actually the right solution?`
2. `Why are AsterPay's operating costs growing faster than revenue?`
3. `Which portfolio company should we prioritize and why?`
4. `How would you prove that an AsterPay AI pilot creates business value?`
5. `Why should we NOT invest in AI for Northstar Wealth?`

## Python libraries
- **Streamlit** — web application/dashboard user interface.
- **Pandas** — tabular data loading, filtering and calculations.
- **Plotly** — interactive visualizations.
- **scikit-learn** — TF-IDF (Term Frequency–Inverse Document Frequency) and cosine similarity for lightweight local RAG.
- **OpenAI Python SDK (Software Development Kit)** — optional LLM reasoning and structured JSON responses.
- **python-dotenv** — loads environment variables such as the API key.

## Architecture
Question → Intent Classification → Company Resolution → Tool Selection → Deterministic Metrics → Question-aware RAG → Root Cause / AI Suitability → LLM or Local Fallback → Recommendation + KPI (Key Performance Indicator) + Controls.

## Important design choice
The LLM never calculates trusted financial metrics and never autonomously makes high-impact payment decisions. Python tools calculate metrics; the model reasons over those outputs. Human approval remains in the control layer.

## Disclaimer
All portfolio companies and data in this repository are synthetic demonstration data. This prototype is designed to demonstrate AI-driven portfolio value creation and is not investment advice.
