# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 07:51:18 2025

@author: Joe
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Initialize the ChatAnthropic model
# You can specify the model name (e.g., "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240229")
# If ANTHROPIC_API_KEY is set as an environment variable, it will be automatically picked up.
# Otherwise, you can pass it as a named parameter: anthropic_api_key="your_anthropic_api_key_here"
API_KEY = os.getenv('ANTHROPIC_API_KEY')
model = ChatAnthropic(model_name="claude-sonnet-4-5", temperature=0.7, anthropic_api_key=API_KEY)

# Create a HumanMessage
message = HumanMessage(content="What is the capital of France?")

# Invoke the model
response = model.invoke([message])

# Print the model's response
print(response.content)