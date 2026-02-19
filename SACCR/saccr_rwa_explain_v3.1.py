"""
Second demo version with missing value func
"""

import os
import pandas as pd
import inspect
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass
import contextlib
import io
import random
import numpy as np

import saccr_cal

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
# from langgraph.checkpoint.memory import InMemorySaver

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

# Set up memory
# checkpointer = InMemorySaver()

@st.cache_data
def compare_ead_cached(df_old, df_new):

    data_old, data_new = df_old, df_new
    summary_old = data_old.reset_index()
    summary_new = data_new.reset_index()
    
    merged = summary_old.merge(summary_new, on="ID", suffixes=("_old", "_new"), how="outer")

    for col in ["Replacement_Cost","PFE_addon","multiplier","EAD", "System_EAD"]:
        merged[f"{col}_Δ"] = merged.get(f"{col}_new", 0) - merged.get(f"{col}_old", 0)
    return merged

@st.cache_data
def summarize_inputs_cached(df_old, df_new):
    drivers = []
    for label, df in [("old", df_old), ("new", df_new)]:

        df_clean = df.convert_dtypes()
        # print(df_clean.dtypes)
        numeric_cols = df_clean.select_dtypes('number').columns.tolist()

        desc = df_clean[numeric_cols].describe().T.round(2)
        desc["dataset"] = label
        drivers.append(desc)
    return pd.concat(drivers)

@st.cache_data
def summarize_inputs_cached_all(dict_old, dict_new):
    drivers = []
    for label, res in [("old", dict_old), ("new", dict_new)]:
        df = pd.DataFrame()
        for k, v in res.items():
            v["NS_ID"] = k
            df = pd.concat([df, v], ignore_index=True)

        df_clean = df.convert_dtypes()
        # print(df_clean.dtypes)
        numeric_cols = df_clean.select_dtypes('number').columns.tolist()

        desc = df_clean[numeric_cols].describe().T.round(2)
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

@st.cache_data
def summarize_input_diff_saccr(data_old, data_new, tab_name, id_cols, exclude_cols=None):
    
    df1 = pd.read_excel(data_old, sheet_name=tab_name) #if isinstance(data_old, str) else data_old.copy()
    df2 = pd.read_excel(data_new, sheet_name=tab_name) #if isinstance(data_new, str) else data_new.copy()
    
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

@st.cache_data
def validate_data(res_all, sys_val_col, cal_val_col, id_cols):
    res_trade = pd.DataFrame()
    for k, v in res_all.items():
        df = v.loc[v["Type"] == "Trade"]
        df["Netting_ID"] = k
        res_trade = pd.concat([res_trade, df], axis=0)
    res_trade.reset_index(drop=True, inplace=True)
    
    diff_mask = (abs(res_trade[sys_val_col] - res_trade[cal_val_col])>0.01) & ~(res_trade[sys_val_col].isna() & res_trade[cal_val_col].isna())

    if not diff_mask.any():
        return pd.DataFrame(columns=list(id_cols) + [cal_val_col, "System Value", "Calculated Value"])    
    
    results = pd.concat(
        [
            res_trade.loc[diff_mask, id_cols].reset_index(drop=True),
            pd.DataFrame({
                "column": cal_val_col,
                "System Value": res_trade.loc[diff_mask, sys_val_col].values,
                "Calculated Value": res_trade.loc[diff_mask, cal_val_col].values,
            })
        ],
        axis=1
    )
    
    return results

    
def summarize_missing_data(df_old, df_new):
    #Currently only handling missing currency codes
    columns = ["Netting Set ID", "Source Txns ID", "Date","Market Value", "ISO CCY", "Principal", "Prin CCY", "Agr Settlement Ccy code"]
    df = pd.concat([df_old[columns], df_new[columns]], ignore_index=True)

    flag_iso_null = (df["Market Value"].notna()) & (df["Market Value"] != 0) & df["ISO CCY"].isna()

    flag_prin_null = (df["Principal"].notna()) & (df["Principal"] != 0) & df["Prin CCY"].isna()

    flag_settle_null = df["Agr Settlement Ccy code"].isna()

    # Build the Reason column
    reasons = []
    for iso_flag, prin_flag, settle_flag in zip(flag_iso_null, flag_prin_null, flag_settle_null):
        tags = []
        if iso_flag:
            tags.append("ISO CCY null when Market Value not null/non-zero; Defaulted to USD")
        if prin_flag:
            tags.append("Prin CCY null when Principal not null/non-zero; Defaulted to USD")
        if settle_flag:
            tags.append("Agr Settlement Ccy code null; Defaulted to USD")
        reasons.append(";".join(tags))

    df["Reason"] = reasons

    # Final mask
    mask = flag_iso_null | flag_prin_null | flag_settle_null

    # Output dataframe
    df_issues = df.loc[
        mask,
        ["Netting Set ID", "Source Txns ID", "Date", "Reason"]
    ]

    return df_issues

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


def ai_agent(code_text, driver_summary, ead_deltas, csa_difference, trade_difference, CSA_inputs, supervisory_delta_check, mpor_check, llm_model):
    
    SYSTEM_PROMPT = f"""
    You are an expert financial risk analyst.

    Assume the following:
    - There are no data quality issues and formatting issues since all data is cleaned before getting processed.
    - Netting set margin status is in CSA input data {CSA_inputs}
    - If Margin Dispute occurs, holding period (MPOR) impact should follow the rule: https://www.ecfr.gov/current/title-12/part-217/section-217.132#p-217.132(b)(2)(ii)(A)(5)(ii)
    
    You are given:
    - Python code for EAD calculation
    - Comparison deltas between old and new input data and results
    - These are the rules for calculating EAD under SACCR approach: https://www.ecfr.gov/current/title-12/part-217/section-217.132#p-217.132(c)

    EAD Calculation Code:
    {code_text}
    
    CSA and Trade Input Difference Summary: 
    {csa_difference}
    {trade_difference}

    Input Data Summary (key drivers):
    {driver_summary}

    EAD comparison (with deltas) between old and new results:
    {ead_deltas}

    Now, write a clear narrative explaining:
    - What changed between old and new datasets
    - Which components drove the EAD differences. Hint: Look for the key drivers in the Netting Set level Data Summary. Then check the input difference summary file to see what changed in the input file and combine the change with EAD Calculation Code logic (but do not show code text in the output) to see how the change in input file result in different EAD value.
    - Any specific patterns or anomalies you can infer
    - When asked to verify the EAD, compare the EAD column and the System_EAD column to see if the difference is greater than 0.01% of the EAD value. If so, flag it as a potential issue.
    - When asked to validate the Supervsory Delta, compare the sysem delta value with the calculated value {supervisory_delta_check}. Hint: transaction type can be determined by calcualted delta vaule. Do not mention call or put.
    - When asked to validate the MPOR, compare the sysem MPOR value with the calculated MPOR value {mpor_check}. Hint: MPOR calculation logic is embeded in the EAD Calculation Code 
    """

    model = init_chat_model(
        llm_model,
        temperature=0,
        max_tokens=1000
    )

    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        # tools=tools,
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

############# Streamlit UI ########################################################
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
    file_old = st.file_uploader("📂 Upload OLD dataset (xlsx)", type=["xlsx"])
with col2:
    file_new = st.file_uploader("📂 Upload NEW dataset (xlsx)", type=["xlsx"])

if file_old and file_new:
    with st.spinner("Processing data..."):
       
        ns_old = saccr_cal.feed_data_excel(file_old)
        ns_new = saccr_cal.feed_data_excel(file_new)
        
        res_old, df_old = saccr_cal.Cal_EAD(ns_old)
        res_new, df_new = saccr_cal.Cal_EAD(ns_new)
        
        merged = compare_ead_cached(df_old, df_new)
        driver_summary = summarize_inputs_cached_all(res_old, res_new) 
        
        CSA_inputs = pd.read_excel(file_old, sheet_name="csa_inputs")
        
        id_cols = ['Netting_ID', 'Margin_ID']
        csa_difference = summarize_input_diff_saccr(file_old, file_new, "csa_inputs", id_cols, [])

        id_cols = ['Netting_ID', 'Trade_ID']
        trade_difference = summarize_input_diff_saccr(file_old, file_new, "trades_inputs", id_cols, [])
        
        code_text = inspect.getsource(saccr_cal)
        ead_deltas = merged.to_string()

        supervisory_delta_check = validate_data(res_old, "System_Delta", "Supervisory_Delta", id_cols)
        
        mpor_check = validate_data(res_old, "System_MPOR", "Calculated_MPOR", id_cols) 

        # tools = build_tools(df_old, df_new)

        if st.session_state.agent is None:
            st.session_state.agent = ai_agent(code_text, driver_summary, ead_deltas, csa_difference, trade_difference, CSA_inputs, supervisory_delta_check, mpor_check, llm_model)

        # st.subheader("📈 EAD Summary")
        # st.dataframe(
        #     merged.reset_index(drop=True), 
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
    
    
# if __name__ == '__main__':
# #     code_text = inspect.getsource(saccr_cal)
# #     print(code_text)
#     ns_old = saccr_cal.feed_data_excel('inputs_data1_v4.xlsx')
#     res_all_old, df_old = saccr_cal.Cal_EAD(ns_old)
    
#     ns_new = saccr_cal.feed_data_excel('inputs_data2_v4.xlsx')
#     res_all_new, df_new = saccr_cal.Cal_EAD(ns_new) 
    
#     # # driver_summary = summarize_inputs_cached(df_old, df_new)
#     # driver_summary = summarize_inputs_cached_all(res_all_old, res_all_new)   
    
#     # id_cols = ['Netting_ID', 'Margin_ID']
#     # csa_diff = summarize_input_diff_saccr('inputs_data1_v1.xlsx', 'inputs_data2_v1.xlsx', "csa_inputs", id_cols, [])

#     # id_cols = ['Netting_ID', 'Trade_ID']
#     # trade_diff = summarize_input_diff_saccr('inputs_data1_v1.xlsx', 'inputs_data2_v1.xlsx', "trades_inputs", id_cols, [])
    
#     id_cols = ['Netting_ID', 'Trade_ID']
#     res_trade = validate_data(res_all_old, "System_Delta", "Supervisory_Delta", id_cols)
    
#     mpor_check = validate_data(res_all_old, "System_MPOR", "Calculated_MPOR", id_cols)
    
#     print(mpor_check)
    