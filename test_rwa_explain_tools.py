## with tools, no data in system prompt

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass
import contextlib
import io
import random
import numpy as np

from rwa_calc import CalcEAD

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv(find_dotenv())

llm_model = "anthropic:claude-sonnet-4-5" # "openai:gpt-4o"
SEED = 32
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Define context schema
@dataclass
class Context:
    user_id: str

# Define response format
@dataclass
class ResponseFormat:
    response: str

@st.cache_data
def compare_ead_cached(df_old, df_new):
    data_old, data_new = CalcEAD(df_old), CalcEAD(df_new)
    summary_old = data_old.df_rwa_summary.reset_index()
    summary_new = data_new.df_rwa_summary.reset_index()

    merged = summary_old.merge(summary_new, on="Netting Set ID", suffixes=("_old", "_new"), how="outer")

    for col in ["Exposure","Collateral","Sec Addon","FX Addon","EAD"]:
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

@st.cache_data
def summarize_input_diff(df_old, df_new, id_cols, exclude_cols=None):
    
    df1 = pd.read_csv(df_old) if isinstance(df_old, str) else df_old.copy()
    df2 = pd.read_csv(df_new) if isinstance(df_new, str) else df_new.copy()
    
    exclude_cols = (exclude_cols or []) + id_cols
 
    df1 = df1.sort_values(id_cols).reset_index(drop=True)
    df2 = df2.sort_values(id_cols).reset_index(drop=True)

    diff_mask = (df1 != df2) & ~(df1.isna() & df2.isna())

    diff_list = []
    for col in df1.columns:
        if col in exclude_cols:
            continue
        changed_rows = diff_mask[col]
        if changed_rows.any():
            temp = pd.concat([df1.loc[changed_rows, id_cols],
                              pd.DataFrame({
                                  'column': col,
                                  'old_value': df1.loc[changed_rows, col],
                                  'new_value': df2.loc[changed_rows, col]
                              })],
                             axis=1)
            diff_list.append(temp)

    if not diff_list:
        return pd.DataFrame(columns=id_cols + ['column', 'old_value', 'new_value'])

    return pd.concat(diff_list, ignore_index=True)

def build_tools(df_old, df_new, merged, input_diff, driver_summary):

    @tool
    def get_netting_set_summary(netting_set_id: str) -> str:
        """Return EAD and deltas for a specific netting set."""
        row = merged[merged["Netting Set ID"] == netting_set_id]
        if row.empty:
            return "Netting set not found."
        return row.to_string()

    @tool
    def get_changed_rows(netting_set_id: str) -> str:
        """Return only the changed inputs for a specific netting set."""
        subset = input_diff[input_diff["Netting Set ID"] == netting_set_id]
        if subset.empty:
            return "No changed rows for this netting set."
        return subset.to_string()

    @tool
    def get_driver_summary() -> str:
        """Return statistical summary of inputs (old vs new)."""
        return driver_summary.to_string()

    @tool
    def preview_data(which: str, n: int = 5) -> str:
        """Preview top rows from 'old' or 'new' dataset."""
        df = df_old if which.lower() == "old" else df_new
        return df.head(n).to_string()

    return [get_netting_set_summary, get_changed_rows, get_driver_summary, preview_data]


def ai_agent(tools,llm_model):

    SYSTEM_PROMPT = f"""
    You are an experienced financial risk analyst specializing in EAD under the collateral haircut approach. 
    
    You have access to tools that let you inspect:
    - Netting-set-level deltas
    - Changed input rows
    - Old/New dataset previews
    - Driver statistics

    KEEP RESPONSES:
    - Concise but complete
    - Explain *why* EAD changed using tool data
    - Never claim "I don't have access to data" — use the tools

    When necessary, call a tool to retrieve precise data.
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
        checkpointer=None
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

st.set_page_config(page_title="EAD Variance Analyzer", layout="wide")
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

        id_columns = ['Netting Set ID', 'Source Txns ID', 'Security ID']
        input_diff = summarize_input_diff(df_old, df_new, id_columns)

        tools = build_tools(df_old, df_new, merged, input_diff, driver_summary)

        if st.session_state.agent is None:
            st.session_state.agent = ai_agent(tools, llm_model)

        # st.subheader("📈 EAD Summary")
        # st.dataframe(
        #     input_difference,
        #     # merged.reset_index(drop=True), 
        #     use_container_width=True, 
        #     hide_index=True
        # )

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
                render_response(msg["content"])

        if user_q := st.chat_input("Enter a question..."):
            with st.chat_message("user"):
                st.markdown(user_q)
            st.session_state.chat_history.append({"role": "user", "content": user_q})

            with st.chat_message("assistant"):
                with st.spinner("Analyzing and generating explanation..."):
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf): 
                        response = st.session_state.agent.invoke(
                            {"messages": [{"role": "user", "content": user_q}]},
                            config={"configurable": {"thread_id": st.session_state.thread_id}},
                            context=Context(user_id="1")
                        )

                explanation = response["structured_response"].response
                render_response(explanation)
                st.session_state.chat_history.append({"role": "assistant", "content": explanation})
            placeholder = st.empty()
            placeholder.markdown("")

else:
    st.info("👆 Please upload both old and new CSV files to begin.")