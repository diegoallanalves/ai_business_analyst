import streamlit as st
import plotly.express as px
from services.tools import load_portfolio, calculate_opportunity
from services.agent import investigate

st.set_page_config(page_title="AI Value Creation Agent", layout="wide")
st.title("AI Value Creation Agent")
st.caption("Question-aware Agentic AI prototype for financial-services portfolio value creation")
df = calculate_opportunity(load_portfolio())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio companies", len(df)); c2.metric("Highest opportunity", f"{df.opportunity_score.max():.1f}"); c3.metric("Average automation", f"{df.automation_pct.mean():.0f}%"); c4.metric("Average data quality", f"{df.data_quality.mean():.0f}/100")
chart_left, chart_right = st.columns([1.65, 1])
with chart_left:
    st.subheader("Portfolio opportunity map")
    scatter = px.scatter(df, x="revenue_growth_pct", y="cost_growth_pct", text="company", hover_data=["margin_pct", "automation_pct", "data_quality", "complaints", "opportunity_score"])
    scatter.update_traces(textposition="top center", marker_size=14)
    scatter.update_layout(xaxis_title="Revenue growth %", yaxis_title="Cost growth %", height=390, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(scatter, use_container_width=True)
with chart_right:
    st.subheader("Opportunity share")
    pie = px.pie(df, names="company", values="opportunity_score", hole=0.48, hover_data=["sector", "data_quality", "automation_pct"])
    pie.update_traces(textinfo="percent+label"); pie.update_layout(height=390, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(pie, use_container_width=True)
    st.caption("Relative share of the prototype opportunity scores — not investment allocation.")
left, right = st.columns([1, 1.5])
with left:
    company = st.selectbox("Select portfolio company", df.company.tolist(), help="Used as the default company only when your question does not name a company.")
    st.caption("If your question names another portfolio company, the Agent automatically uses the company in the question.")
    question = st.text_area("Ask the AI Agent", value="Suppose AsterPay has 92 complaints. How do you know AI is actually the right solution?", height=100)
    run = st.button("Run Agent Investigation", type="primary", use_container_width=True)
with right:
    st.subheader("Agent capabilities")
    st.write("1. **Portfolio Comparison Tool** — compares companies")
    st.write("2. **Financial Analysis Tool** — trusted deterministic metrics")
    st.write("3. **Question-aware RAG (Retrieval-Augmented Generation) Search Tool** — retrieves relevant company evidence")
    st.write("4. **Root Cause / AI (Artificial Intelligence) Suitability Tool** — checks whether AI is justified")
    st.write("5. **Recommendation / Business Value Tool** — proposes next action and KPI (Key Performance Indicator)")
if run:
    result = investigate(question, company); st.divider(); st.subheader("Agent investigation")
    if result["company_source"] == "question" and result["company"] != company:
        st.info(f"Company override detected: the dropdown selected **{company}**, but your question names **{result['company']}**. The Agent is using **{result['company']}**.")
    elif result["company_source"] == "portfolio_ranking": st.info(f"Portfolio-wide question detected. The Agent selected **{result['company']}** from the portfolio comparison.")
    else: st.caption(f"Using dropdown company: **{result['company']}**")
    cols = st.columns(len(result["tools"]))
    for i, (col, tool) in enumerate(zip(cols, result["tools"]), 1): col.success(f"Step {i}\n\n{tool}")
    st.caption(f"Detected intent: **{result['intent']}** | Answer mode: **{'LLM (Large Language Model) + tools' if result['used_llm'] else 'deterministic local fallback'}**")
    st.subheader(f"Selected opportunity: {result['company']}"); m = result["metrics"]
    a,b,c,d=st.columns(4); a.metric("Opportunity score",f"{m['opportunity_score']}/100"); b.metric("Margin",f"{m['margin_pct']:.1f}%"); c.metric("Cost growth",f"{m['cost_growth_pct']:.1f}%"); d.metric("Complaints",int(m["complaints"]))
    with st.expander("RAG (Retrieval-Augmented Generation) evidence", expanded=True):
        if result["rag"]:
            for item in result["rag"]: st.write(f"**{item['source']}** · relevance {item['score']}"); st.write(item["text"])
        else: st.write("No relevant company context retrieved.")
    st.subheader("Agent answer"); ans=result["answer"]
    for heading,key in [("Decision","decision"),("Why","why"),("What should we investigate next?","next_investigation"),("AI pilot — only if justified","ai_pilot"),("Business value / KPI (Key Performance Indicator)","kpi"),("Risk and control","risk_control"),("Suggested approach","suggested_approach")]: st.markdown(f"### {heading}"); st.write(ans[key])
    st.info("Agentic workflow: question → intent → company resolution → tool selection → trusted metrics → question-aware RAG (Retrieval-Augmented Generation) → root-cause / suitability assessment → answer → KPI (Key Performance Indicator) + controls.")
st.divider(); st.caption("Synthetic demonstration data. Prototype for AI-driven portfolio value creation; not investment advice.")
