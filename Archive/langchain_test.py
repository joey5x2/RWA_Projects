# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 07:51:18 2025

@author: Joe
"""

import os
import streamlit as st
from langchain_core.prompts import PromptTemplate


# from langchain.globals import set_debug
# set_debug(True)

from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

print(OPENAI_API_KEY)

prompt_template = PromptTemplate(
    input_variables = ["country", "number", "topic", "table_name"],
    template = """Welcome to the Basel rule inquiry!
    1. List the {number} most important {country} Basel rules for {topic} 
    2. Display the CHA formula for exposure amount calculation
    3. Print out the requested {table_name} table in full table format
    Please refer to this reference link: https://www.ecfr.gov/current/title-12/chapter-II/subchapter-A/part-217/subpart-D/subject-group-ECFR90182ef648cc7a4/section-217.37
    Hope this helps!
    """
    )


llm = ChatOpenAI(model='gpt-4o',api_key=OPENAI_API_KEY)

# st.title("Ask a question for AI")

# question = "what is the largest city in US?"
# question = st.text_input("Enter queiston here:")

# if question:
#     response = llm.invoke(question)
#     st.write(response.content)

# print(response.content)

st.title("Basel Rule Inquiry")

country = st.text_input("Country of interest:")
number = st.number_input("Number of rules:", min_value=1, max_value=10)
topic = st.text_input("Topic if interest:") 
table_name = st.text_input("Table to request:")


if country:
    response = llm.invoke(prompt_template.format(country=country,
                                                 number=number,
                                                 topic=topic,
                                                 table_name=table_name))
    st.write(response.content)
    
# # --------- ollama ----------------
# from langchain_community.chat_models import ChatOllama
# # from langchain_core.prompts import ChatPromptTemplate


# # prompt = ChatPromptTemplate.from_messages([("system","You are a helpful AI Assistant."),
# #                                            ("user","{question}")
# #     ])

# prompt_template = PromptTemplate(
#     input_variables = ["country", "number", "topic", "table_name"],
#     template = """Welcome to the Basel rule inquiry!
#     The {number} most important {country} Basel rules for {topic} are listed below:
#     The requested rule {table_name} table is shown as follows:
#     Hope this helps!
#     """
#     )

# llm = ChatOllama(model="llama3")

# st.title("Basel Rule Inquiry")
# country = st.text_input("Country of interest:")
# number = st.number_input("Number of rules:", min_value=1, max_value=10)
# topic = st.text_input("Topic if interest:") 
# table_name = st.text_input("Table to request:")


# # chain = prompt | llm

# # question = "what is the largest city in US?"

# # response = chain.invoke(question)

# if country:
#     response = llm.invoke(prompt_template.format(country=country,
#                                                  number=number,
#                                                  topic=topic,
#                                                  table_name=table_name))
#     st.write(response.content)
    
# # print(response.content)
