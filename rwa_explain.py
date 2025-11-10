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
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool

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
def compare_ead(df_old, df_new):
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

def summarize_inputs(df_old, df_new):
    drivers = []
    for label, df in [("old", df_old), ("new", df_new)]:
        numeric_cols = df.select_dtypes("number").columns.tolist()
        desc = df[numeric_cols].describe().T.round(2)
        desc["dataset"] = label
        drivers.append(desc)
    return pd.concat(drivers)

def ai_agent(code_text,input_data_old,input_data_new,driver_summary,ead_deltas,llm_model):

    input_data_old_text = input_data_old.to_csv(index=False)
    input_data_new_text = input_data_new.to_csv(index=False)

    SYSTEM_PROMPT = f"""
    You are an expert financial risk analyst.

    You are given:
    1. The Python code that calculates Exposure at Default (EAD).
    2. Two sets of input data in full.
    3. Two sets of input data (old vs new) summarized statistically.
    4. The calculated EAD summaries for both datasets, and the numeric deltas.

    Your job:
    - Explain what likely caused the increase or decrease in EAD.
    - Identify which input variables (e.g., Market Value, Principal, Haircut, MPOR, Buy/Sell Indicator) changed and how they influenced the result.
    - Focus on concrete quantitative drivers, not vague explanations.
    - Use reasoning based on the code logic (how collateral, exposure, and add-ons interact).

    EAD Calculation Code:
    {code_text}

    Here are two datasets in full:
    OLD dataset:
    {input_data_old_text}

    NEW dataset:
    {input_data_new_text}

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
        temperature=0.1,
        max_tokens=1500
    )

    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
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

        merged = compare_ead(df_old, df_new)
        driver_summary = summarize_inputs(df_old, df_new)
        code_text = inspect.getsource(rwa_calc.CalcEAD)
        ead_deltas = merged.to_string()

        if st.session_state.agent is None:
            st.session_state.agent = ai_agent(code_text, df_old, df_new, driver_summary, ead_deltas, llm_model)

        st.subheader("📈 EAD Summary Comparison")
        st.dataframe(
            merged.reset_index(drop=True).style.format("{:,.0f}"), 
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