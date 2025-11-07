import os
from typing import Optional

import streamlit as st
import pandas as pd

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

from rwa_calc_revised import CalcEAD  # ensure this is in the same folder
import inspect
# ====================================================
# Setup
# ====================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
code_text = inspect.getsource(CalcEAD)
st.set_page_config(
    page_title="RWA EAD Driver Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("📊 RWA EAD Driver Analysis (Netting Set & Trade-Level Drivers)")
st.caption(
    "Flow:\n"
    "1) Upload OLD and NEW portfolios.\n"
    "2) App runs CalcEAD to get netting-set EAD.\n"
    "3) EAD is merged back to trades.\n"
    "4) Netting-set EAD deltas are computed explicitly.\n"
    "5) A pandas DataFrame agent (tool-calling, Python-enabled) answers questions and can drill to transaction level."
)

# ====================================================
# Session State
# ====================================================
for key in [
    "df_old_raw",
    "df_new_raw",
    "df_old_ead",
    "df_new_ead",
    "sum_old",
    "sum_new",
    "netting_diff",
    "agent",
    "chat_history",
]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.chat_history is None:
    st.session_state.chat_history = []


# ====================================================
# Helpers
# ====================================================
def run_calcead_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run CalcEAD(df) -> df_rwa_summary and normalize EAD column to 'EAD_std'.

    Supports:
      - 'Reported EAD'
      - 'EAD'
    """
    calc = CalcEAD(df)
    if not hasattr(calc, "df_rwa_summary"):
        raise ValueError("CalcEAD has no attribute 'df_rwa_summary'.")

    out = calc.df_rwa_summary
    if not isinstance(out, pd.DataFrame):
        raise ValueError("CalcEAD.df_rwa_summary is not a DataFrame.")

    # Ensure Netting Set ID column
    if "Netting Set ID" not in out.columns and out.index.name == "Netting Set ID":
        out = out.reset_index()
    if "Netting Set ID" not in out.columns:
        raise ValueError("CalcEAD summary missing 'Netting Set ID' column.")

    # Pick EAD column
    ead_col = None
    if "Reported EAD" in out.columns:
        ead_col = "Reported EAD"
    elif "EAD" in out.columns:
        ead_col = "EAD"
    if ead_col is None:
        raise ValueError("CalcEAD summary missing 'Reported EAD' or 'EAD'.")

    out = out.copy()
    out["EAD_std"] = out[ead_col]
    return out


def merge_ead_back(trades: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """
    Merge CalcEAD summary back to trade rows via 'Netting Set ID'.

    Result = original columns + summary columns (incl. EAD_std).
    """
    if "Netting Set ID" not in trades.columns:
        raise ValueError("Trades file missing 'Netting Set ID' column.")

    merged = trades.merge(summary, on="Netting Set ID", how="left")
    return merged


def build_netting_diff(sum_old: pd.DataFrame, sum_new: pd.DataFrame) -> pd.DataFrame:
    """
    Netting-set level EAD deltas
    based on EAD_std.
    """
    cols = sum_old.columns.intersection(sum_new.columns).tolist()
    cols_rename_old = [ f"{c}_old" if c != "Netting Set ID" else c for c in cols ]
    cols_rename_new = [ f"{c}_new" if c != "Netting Set ID" else c for c in cols ]
    so = sum_old[cols].rename(columns=dict(zip(cols, cols_rename_old)))
    sn = sum_new[cols].rename(columns=dict(zip(cols, cols_rename_new)))

    diff = so.merge(sn, on="Netting Set ID", how="outer")
    diff["Reported EAD_old"] = diff["Reported EAD_old"].fillna(0.0)
    diff["Reported EAD_new"] = diff["Reported EAD_new"].fillna(0.0)
    diff["EAD_delta"] = diff["Reported EAD_new"] - diff["Reported EAD_old"]

    diff = diff.sort_values("EAD_delta", ascending=False).reset_index(drop=True)
    return diff


def build_agent() -> Optional[object]:
    """
    Create a pandas DataFrame agent with:

      df_list[0] = df_old_ead     (OLD trades + EAD_std + CalcEAD cols)
      df_list[1] = df_new_ead     (NEW trades + EAD_std + CalcEAD cols)
      df_list[2] = netting_diff   (Netting Set ID, EAD_old, EAD_new, EAD_delta)

    Uses:
      - allow_dangerous_code=True for Python REPL over DataFrames
      - agent_type="openai-tools" for robust tool-calling (no brittle ReAct parsing)
      - handle_parsing_errors=True so minor format issues are retried/handled
    """
    if (
        st.session_state.df_old_ead is None
        or st.session_state.df_new_ead is None
        or st.session_state.netting_diff is None
    ):
        return None

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not set.")
        return None

    llm = ChatOpenAI(
        temperature=0,
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
    )

    dfs = [
        st.session_state.df_old_ead,
        st.session_state.df_new_ead,
        st.session_state.netting_diff,
    ]

    agent = create_pandas_dataframe_agent(
        llm,
        dfs,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        agent_type="openai-tools",
    )

    return agent


def agent_hint() -> str:
    """
    Lightweight hint to steer the agent without forcing invalid formats.
    """
    return (
        "\n\nContext for you:\n"
        "- df_list[0]: OLD trades with 'Netting Set ID' and EAD_std.\n"
        "- df_list[1]: NEW trades with 'Netting Set ID' and EAD_std.\n"
        "- df_list[2]: Netting-set EAD deltas with 'Netting Set ID', 'EAD_delta' and  direct calculate inputs.\n"
        f"Reported EAD_old and Reported EAD_new are derived from:\n{code_text}"
        

        "When comparing changes, ALWAYS read df_list[2] to find top movers, "
        "Then explain the drivers by looking at the changes in old vs new difference in df_list[2] fields."
        "Thirdly, filter df_list[0] and df_list[1] by 'Netting Set ID' to explain trade-level drivers.\n"
        "Return final answers in natural language (with tables where useful), not as raw planning text."

    )


def safe_show_fallback(e: Exception):
    """
    If the agent chokes on parsing, fall back to a deterministic top-movers view
    so the user still gets something useful.
    """
    msg = str(e)
    if "OUTPUT_PARSING_FAILURE" in msg or "Could not parse LLM output" in msg:
        nd = st.session_state.netting_diff
        if isinstance(nd, pd.DataFrame):
            top = nd.head(10)
            st.warning(
                "The LLM agent had trouble formatting its answer, "
                "so here's a direct view of the top netting set EAD deltas instead."
            )
            st.dataframe(top, use_container_width=True)
        else:
            st.error("Agent parsing failed and no netting_diff is available.")
    else:
        st.error(f"Agent error: {e}")


# ====================================================
# File upload + EAD pipeline
# ====================================================
col1, col2 = st.columns(2)

with col1:
    old_file = st.file_uploader("OLD portfolio CSV", type=["csv"], key="old_csv")
    if old_file is not None:
        try:
            df_old = pd.read_csv(old_file)
            st.session_state.df_old_raw = df_old

            sum_old = run_calcead_summary(df_old)
            df_old_ead = merge_ead_back(df_old, sum_old)

            st.session_state.sum_old = sum_old
            st.session_state.df_old_ead = df_old_ead

            st.success(
                f"OLD loaded. Trades: {df_old.shape[0]}, Netting sets: {sum_old['Netting Set ID'].nunique()}"
            )
            with st.expander("Preview OLD + EAD_std", expanded=False):
                st.dataframe(df_old_ead.head(50), use_container_width=True)
            #with st.expander("Preview sum_old", expanded=False):
            #   st.dataframe(sum_old.head(50), use_container_width=True)
        except Exception as e:
            st.error(f"Error computing OLD EAD via CalcEAD: {e}")

with col2:
    new_file = st.file_uploader("NEW portfolio CSV", type=["csv"], key="new_csv")
    if new_file is not None:
        try:
            df_new = pd.read_csv(new_file)
            st.session_state.df_new_raw = df_new

            sum_new = run_calcead_summary(df_new)
            df_new_ead = merge_ead_back(df_new, sum_new)

            st.session_state.sum_new = sum_new
            st.session_state.df_new_ead = df_new_ead

            st.success(
                f"NEW loaded. Trades: {df_new.shape[0]}, Netting sets: {sum_new['Netting Set ID'].nunique()}"
            )
            with st.expander("Preview NEW + EAD_std", expanded=False):
                st.dataframe(df_new_ead.head(50), use_container_width=True)
            #with st.expander("Preview sum_new", expanded=False):
            #    st.dataframe(sum_new.head(50), use_container_width=True)
        except Exception as e:
            st.error(f"Error computing NEW EAD via CalcEAD: {e}")

# Build netting_diff once both summaries exist
if (
    st.session_state.get("sum_old") is not None
    and st.session_state.get("sum_new") is not None
):
    try:
        netting_diff = build_netting_diff(
            st.session_state.sum_old,
            st.session_state.sum_new
        )
        st.session_state.netting_diff = netting_diff
        with st.expander("Netting Set EAD Differences (OLD vs NEW)", expanded=True):
            st.dataframe(netting_diff, use_container_width=True)
    except Exception as e:
        st.error(f"Error building netting-set EAD diff: {e}")

# ====================================================
# Build agent once data ready
# ====================================================
if (
    st.session_state.df_old_ead is not None
    and st.session_state.df_new_ead is not None
    and st.session_state.netting_diff is not None
    and st.session_state.agent is None
):
    st.session_state.agent = build_agent()

agent = st.session_state.agent

# ====================================================
# Chat UI
# ====================================================
st.markdown("---")
st.subheader("Ask about EAD changes & transaction-level drivers")

if agent is None:
    st.info("Upload both OLD and NEW portfolios (with successful CalcEAD runs) to start the analysis.")
else:
    # replay history
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    user_q = st.chat_input(
        "Example: Identify the top 2 netting sets by EAD_delta and explain trade-level drivers."
    )

    if user_q:
        st.session_state.chat_history.append(("user", user_q))
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing using the pandas agent..."):
                try:
                    full_prompt = user_q + agent_hint()
                    # Use modern invoke API; more robust with agent_type="openai-tools"
                    result = agent.invoke({"input": full_prompt})

                    if isinstance(result, dict):
                        # openai-tools style often returns {"output": "..."} or similar
                        answer = result.get("output") or result.get("final_answer") or str(result)
                    else:
                        answer = str(result)

                    st.markdown(answer)
                    st.session_state.chat_history.append(("assistant", answer))
                except Exception as e:
                    safe_show_fallback(e)
                    st.session_state.chat_history.append(
                        ("assistant", f"(Agent error handled: {e})")
                    )

# ====================================================
# Sidebar Utilities
# ====================================================
st.sidebar.subheader("Utilities")
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun()

if st.sidebar.button("Reset App (forget data & agent)"):
    for key in [
        "df_old_raw",
        "df_new_raw",
        "df_old_ead",
        "df_new_ead",
        "sum_old",
        "sum_new",
        "netting_diff",
        "agent",
        "chat_history",
    ]:
        st.session_state[key] = None
    st.experimental_rerun()

st.sidebar.caption("© 2025 RWA EAD Driver Analysis")
