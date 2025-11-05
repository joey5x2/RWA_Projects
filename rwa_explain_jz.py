# %%
import os
import streamlit as st

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

env = find_dotenv()


# %%
import pandas as pd

def calculate_growth(data: pd.DataFrame):
    data["growth"] = data["rwa"].pct_change() * 100
    return data

# Example data
data = pd.DataFrame({
    "month": ["Jan", "Feb", "Mar", "Apr"],
    "rwa": [1000, 1200, 900, 1600]
})

# %%
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import PromptTemplate


# # Define system prompt
SYSTEM_PROMPT = f"""
Given the following rwa data and growth rates:

{data}

Explain why the rwa increased or decreased between periods.
Consider trends, external factors, or data patterns.

You have access to three tools:
- explain_drivers: use this to explain why number increased or decreased
- get_user_team: use this to get the user's team
- calculate_rwa_impact: use this to calculate rwa impact
"""

print(SYSTEM_PROMPT)

# prompt_template = PromptTemplate(
#     input_variables = ["data_file", "question"],
#     template = """
#     Given the following rwa data and growth rates:
#     {data}
    
#     {question}
    
#     You have access to three tools:
#     - explain_drivers: use this to explain why number increased or decreased
#     - get_user_team: use this to get the user's team
#     - calculate_rwa_impact: use this to calculate rwa impact
#     """
#     )

# question = "Explain why the rwa increased or decreased between periods. Consider trends, external factors, or data patterns."
# %%

# Define context schema
@dataclass
class Context:
    user_id: str

@tool
def explain_drivers(month: str) -> str:
    """Explain RWA drivers."""
    return f"E increased or C decreased in {month}."

@tool
def get_user_team(runtime: ToolRuntime[Context]) -> str:
    """Get user's team."""
    user_id = runtime.context.user_id
    return "FIN" if user_id == "1" else "BUSINESS"

# @tool
# def calculate_rwa_impact(data: pd.DataFrame) -> str:
#     """Calculate rwa impact from input."""
#     data_str = calculate_growth(data).to_string(index=False)
#     return data_str

# Configure model
model = init_chat_model(
    #"anthropic:claude-sonnet-4-5",
    "openai:gpt-4o", 
    # "google_genai:gemini-2.0-flash",
    temperature=0
)

# Define response format
@dataclass
class ResponseFormat:
    response: str
    rwa_impact: float | None = None

# Set up memory
checkpointer = InMemorySaver()

# Create agent
st.title("Variance Analysis Tool")
data_file = st.text_input("Input data file:") 
# df = pd.read_csv(data_file)
question = st.text_input("Ask a quesion:")

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_team, explain_drivers],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# Run agent
config = {"configurable": {"thread_id": "1"}}
if question:
    response = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        context=Context(user_id="1")
    )
    st.write(response['structured_response'])
    



# config = {"configurable": {"thread_id": "1"}}

# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "did RWA increase or decrease from Jan to Feb? explain why"}]},
#     config=config,
#     context=Context(user_id="1")
# )

# print(response['structured_response'])

# # we can continue the conversation using the same `thread_id`.
# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "did RWA increase or decrease from Jan to March?"}]},
#     config=config,
#     context=Context(user_id="1")
# )

# print(response['structured_response'])

# %%
