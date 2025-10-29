# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 07:51:18 2025

@author: Joe
"""

import os
from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

print(OPENAI_API_KEY)

llm = ChatOpenAI(model='gpt-4o',api_key=OPENAI_API_KEY)

question = "what is the largest city in US?"

response = llm.invoke(question)

print(response)