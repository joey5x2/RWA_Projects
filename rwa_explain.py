# %%
import os
import pandas as pd
import inspect
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass

import rwa_calc
from rwa_calc import CalcEAD

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(find_dotenv())

llm_model = "openai:gpt-4o" # "anthropic:claude-sonnet-4-5"

# Define context schema
@dataclass
class Context:
    user_id: str

# Define response format
@dataclass
class ResponseFormat:
    response: str

# Set up memory
checkpointer = InMemorySaver()

# %%
@st.cache_data
def compare_ead_cached(df_old, df_new):
    data_old, data_new = CalcEAD(df_old), CalcEAD(df_new)
    summary_old = data_old.df_rwa_summary.reset_index()
    summary_new = data_new.df_rwa_summary.reset_index()

    merged = summary_old.merge(summary_new, on="Netting Set ID", suffixes=("_old", "_new"), how="outer")

    for col in [
        "Trade Level Exposure",
        "Trade Level Collateral - Scen A",
        "Trade Level Collateral - Scen B",
        "EAD - Scen A",
        "EAD - Scen B",
        "EAD"
    ]:
        merged[f"{col}_Δ"] = merged.get(f"{col}_new", 0) - merged.get(f"{col}_old", 0)
    return merged

@st.cache_data
def summarize_inputs_cached(df_old, df_new):
    drivers = []
    for label, df in [("old", df_old), ("new", df_new)]:
        numeric_cols = df.select_dtypes("number").columns.tolist()
        desc = df[numeric_cols].describe().T.round(2)
        desc["dataset"] = label
        drivers.append(desc)
    return pd.concat(drivers)

def build_tools(df_old, df_new):    
    @tool
    def preview_data(dataset: str, n: int = 5) -> str:
        """Preview the first n rows of the dataset."""
        df = df_old if dataset.lower() == "old" else df_new
        return df.head(n).to_string()

    @tool
    def query_data(dataset: str, query: str, n: int = 10) -> str:
        """Run a pandas query (df.query syntax) safely and return top N rows."""
        df = df_old if dataset.lower() == "old" else df_new
        try:
            result = df.query(query).head(n)
            return result.to_string()
        except Exception as e:
            return f"Query failed: {e}"

    return [preview_data, query_data]

def ai_agent(code_text,driver_summary,ead_deltas,llm_model,tools):

    SYSTEM_PROMPT = f"""
    You are an expert financial risk analyst.

    You can use the following Python tools to explore real datasets:
    - preview_data: to inspect sample rows
    - query_data: to filter data by condition

    You are given:
    - Python code for EAD calculation
    - Summaries of two datasets (old vs new)
    - Comparison deltas between old and new results
    - These are the rules for calculating EAD under collateral haircut approach: https://www.ecfr.gov/current/title-12/chapter-II/subchapter-A/part-217/subpart-D/subject-group-ECFR90182ef648cc7a4/section-217.37

    Always ground your reasoning in what you find through the tools.

    EAD Calculation Code:
    {code_text}

    Input Data Summary (key drivers):
    {driver_summary}

    EAD Comparison Summary (with deltas):
    {ead_deltas}

    Now, write a clear narrative explaining:
    - What changed between old and new datasets
    - Which components drove the EAD differences
    - Any specific patterns or anomalies you can infer
    """

    model = init_chat_model(
        llm_model,
        temperature=0,
        max_tokens=1000
    )

    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        context_schema=Context,
        response_format=ResponseFormat,
        checkpointer=checkpointer
    )

    return agent

def render_response(text: str):
    """Render LLM output nicely, preserving Markdown, math, and spacing."""
    text = text.strip()

    if "$$" in text:
        parts = text.split("$$")
        for i, part in enumerate(parts):
            if i % 2 == 0:
                st.markdown(part)
            else:
                st.latex(part)
    elif "$" in text:
        st.markdown(text.replace("$", r"\$"))
    elif "```" in text:
        st.markdown(text, unsafe_allow_html=True)
    else:
        st.markdown(text)

st.set_page_config(page_title="EAD Delta Analyzer", layout="wide")
st.title("📊 EAD Variance Analyzer with LangChain")
st.write("Upload two datasets (old vs new) to analyze EAD changes and get an AI explanation.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent" not in st.session_state:
    st.session_state.agent = None

col1, col2 = st.columns(2)
with col1:
    file_old = st.file_uploader("📂 Upload OLD dataset (CSV)", type=["csv"])
with col2:
    file_new = st.file_uploader("📂 Upload NEW dataset (CSV)", type=["csv"])

if file_old and file_new:
    with st.spinner("Processing data..."):
        df_old = pd.read_csv(file_old)
        df_new = pd.read_csv(file_new)

        merged = compare_ead_cached(df_old, df_new)
        driver_summary = summarize_inputs_cached(df_old, df_new)
        code_text = inspect.getsource(rwa_calc.CalcEAD)
        ead_deltas = merged.to_string()

        tools = build_tools(df_old, df_new)

        if st.session_state.agent is None:
            st.session_state.agent = ai_agent(code_text, driver_summary, ead_deltas, llm_model, tools)

        st.subheader("📈 EAD Summary Comparison")
        st.dataframe(
            merged.reset_index(drop=True), 
            use_container_width=True, 
            hide_index=True
        )

        st.markdown("### 💬 Conversation Controls")
        if st.button("🔄 Start New Conversation"):
            st.session_state.thread_id = str(int(st.session_state.thread_id) + 1)
            st.session_state.chat_history = []
            st.session_state.agent = None
            st.success("New conversation started!")

        st.divider()
        st.subheader("🤖 Ask a Question")

        # Display prior messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_q := st.chat_input("Enter a question..."):
            with st.chat_message("user"):
                st.markdown(user_q)
            st.session_state.chat_history.append({"role": "user", "content": user_q})

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing and generating explanation..."):
                    response = st.session_state.agent.invoke(
                        {"messages": [{"role": "user", "content": user_q}]},
                        config={"configurable": {"thread_id": st.session_state.thread_id}},
                        context=Context(user_id="1")
                    )

                    explanation = response["structured_response"].response
                    render_response(explanation)
                    st.session_state.chat_history.append({"role": "assistant", "content": explanation})
else:
    st.info("👆 Please upload both old and new CSV files to begin.")