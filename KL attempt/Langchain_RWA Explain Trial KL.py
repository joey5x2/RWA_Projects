import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()  # if using .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print()
import pandas as pd

# DataFrame 1: T-1 data
df1 = pd.DataFrame({
    'Netting Set ID': ['101', '101', '102', '103', '104'],
    'Source Txns ID': ['1', '2', '3', '4', '5'],
    'Exposure': [10001,10020,10300, 14000,14000 ],
    'Eligible Collateral': [100,200,300,400,400],
    'Addon':[99,98,100,136, 136],
    'EAD': [19918,19918,10100,13736,13736 ],
    'RW':[1,1,0.02,1,1],
    'RWA':[19918,19918,202,13736,13736]
})

# DataFrame 2: T data (with some changes)
df2 = pd.DataFrame({
'Netting Set ID': ['101', '101', '102', '103', '104'],
    'Source Txns ID': ['1', '2', '3', '4', '5'],
    'Exposure': [100000001,10020,10300, 14000, 14000],
    'Eligible Collateral': [100,200,300,400, 400],
    'Addon':[99,98,100,136, 136],
    'EAD': [100009918,100009918,10100,13736, 13736],
    'RW':[1,1,1,1,1],
    'RWA':[100009918,100009918,10100,13736,13736]
})

# Name the dataframes for better context in the prompt
df1.name = "df_current"
df2.name = "df_updated"

import os
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=OPENAI_API_KEY)
agent = create_pandas_dataframe_agent(
    llm,
    [df1, df2],
    verbose=True, # Set verbose=True to see the agent's thought process and generated code
    allow_dangerous_code=True # Allow execution of potentially unsafe code
)

prompt = """
In both df_current and df_updated, the column 'EAD' is calculated as Exposure + Eligible Collateral + Addon, the column 'RWA' is calculated as EAD * RW.
Using df_current and df_updated, identify the top 2 netting set with the most change in 'RWA'and provide the driver of the change. 
For driver, drill down to transaction level if needed. 
"""
response = agent.invoke(prompt)
print(response)


"""
Things need to consider to tune:
1. Need to tune AI to have a better understanding if the driver is reasonable or if there could be data issues
    - This could involve providing more examples in the training data or setting thresholds
2. Granulaty of the analysis
    - Should also drill down to transaction level if needed
"""




"""
The top 2 netting sets with the most change in 'RWA' are:

1. Netting Set ID 101: The change is driven by a dramatic increase in 'Exposure' for the transaction with Source Txns ID 1, 
from 10,001 to 100,000,001, leading to a substantial increase in 'EAD' and consequently 'RWA'.

2. Netting Set ID 102: The change is driven by an increase in the 'RW' (Risk Weight) from 0.02 to 1 for the transaction with Source Txns ID 3, 
resulting in a significant increase in 'RWA'.


"""



















# Graveyard Test Output



"""
Output:

To identify the top 2 netting sets with the most change in 'RWA' and provide the driver of the change, we need to:

1. Calculate the change in 'RWA' for each 'Netting Set ID' between the two dataframes.
2. Identify the top 2 netting sets with the largest change in 'RWA'.
3. Determine the driver of the change by analyzing the components: 'Exposure', 'Eligible Collateral', 'Addon', and 'RW'.

Let's proceed with these steps.

Thought: I will calculate the change in 'RWA' for each 'Netting Set ID' and identify the top 2 netting sets with the most change. Then, I will analyze the components to determine the driver of the change.
Action: python_repl_ast
Action Input: 
```python
import pandas as pd

# Sample data for df_current and df_updated
data_current = {
    'Netting Set ID': [101, 101, 102, 103, 104],
    'Source Txns ID': [1, 2, 3, 4, 5],
    'Exposure': [10001, 10020, 10300, 14000, 14000],
    'Eligible Collateral': [100, 200, 300, 400, 400],
    'Addon': [99, 98, 100, 136, 136],
    'EAD': [19918, 19918, 10100, 13736, 13736],
    'RW': [1, 1, 0.02, 1, 1],
    'RWA': [19918, 19918, 202, 13736, 13736]
}

data_updated = {
    'Netting Set ID': [101, 101, 102, 103, 104],
    'Source Txns ID': [1, 2, 3, 4, 5],
    'Exposure': [100000001, 10020, 10300, 14000, 14000],
    'Eligible Collateral': [100, 200, 300, 400, 400],
    'Addon': [99, 98, 100, 136, 136],
    'EAD': [100009918, 100009918, 10100, 13736, 13736],
    'RW': [1, 1, 1, 1, 1],
    'RWA': [100009918, 100009918, 10100, 13736, 13736]
}

df_current = pd.DataFrame(data_current)
df_updated = pd.DataFrame(data_updated)

# Calculate the change in RWA for each Netting Set ID
df_current_grouped = df_current.groupby('Netting Set ID').sum()
df_updated_grouped = df_updated.groupby('Netting Set ID').sum()

df_current_grouped['RWA_change'] = df_updated_grouped['RWA'] - df_current_grouped['RWA']

# Identify the top 2 netting sets with the most change in RWA
top_2_changes = df_current_grouped['RWA_change'].nlargest(2)
top_2_changes
```Netting Set ID
101    199980000
102         9898
Name: RWA_change, dtype: int64The top 2 netting sets with the most change in 'RWA' are Netting Set ID 101 and 102. Now, I need to determine the driver of the change by analyzing the components: 'Exposure', 'Eligible Collateral', 'Addon', and 'RW' for these netting sets.

For Netting Set ID 101, the 'RWA' change is significant, likely due to a large change in 'Exposure'. For Netting Set ID 102, the change is smaller, and I need to check the components to identify the driver.

Let's analyze the components for these netting sets to determine the driver of the change.

Action: python_repl_ast
Action Input: 
```python
# Analyze the components for Netting Set ID 101 and 102
df_current_101 = df_current[df_current['Netting Set ID'] == 101]
df_updated_101 = df_updated[df_updated['Netting Set ID'] == 101]

df_current_102 = df_current[df_current['Netting Set ID'] == 102]
df_updated_102 = df_updated[df_updated['Netting Set ID'] == 102]

# Calculate the sum of components for Netting Set ID 101
components_101_current = df_current_101[['Exposure', 'Eligible Collateral', 'Addon', 'RW']].sum()
components_101_updated = df_updated_101[['Exposure', 'Eligible Collateral', 'Addon', 'RW']].sum()
# Calculate the sum of components for Netting Set ID 101
components_101_current = df_current_101[['Exposure', 'Eligible Collateral', 'Addon', 'RW']].sum()
components_101_updated = df_updated_101[['Exposure', 'Eligible Collateral', 'Addon', 'RW']].sum()

components_101_current = df_current_101[['Exposure', 'Eligible Collateral', 'Addon', 'RW']].sum()
components_101_updated = df_updated_101[['Exposure', 'Eligible Collateral', 'Addon', 'RW']].sum()

components_101_updated = df_updated_101[['Exposure', 'Eligible Collateral', 'Addon', 'RW']].sum()

# Calculate the sum of components for Netting Set ID 102
components_102_current = df_current_102[['Exposure', 'Eligible Collateral', 'Addon', 'RW']].sum()

# Calculate the sum of components for Netting Set ID 102
components_102_current = df_current_102[['Exposure', 'Eligible Collateral', 'Addon', 'RW']].sum()
# Calculate the sum of components for Netting Set ID 102
components_102_current = df_current_102[['Exposure', 'Eligible Collateral', 'Addon', 'RW']].sum()
components_102_current = df_current_102[['Exposure', 'Eligible Collateral', 'Addon', 'RW']].sum()
components_102_updated = df_updated_102[['Exposure', 'Eligible Collateral', 'Addon', 'RW']].sum()

components_101_current, components_101_updated, components_102_current, components_102_updated
```(Exposure               20021.0
Eligible Collateral      300.0
Addon                    197.0
RW                         2.0
dtype: float64, Exposure               100010021
Eligible Collateral          300
Addon                    197.0
RW                         2.0
dtype: float64, Exposure               100010021
Eligible Collateral          300
RW                         2.0
dtype: float64, Exposure               100010021
Eligible Collateral          300
dtype: float64, Exposure               100010021
Eligible Collateral          300
Eligible Collateral          300
Addon                        197
RW                             2
dtype: int64, Exposure               10300.00
Eligible Collateral      300.00
Addon                    100.00
RW                         0.02
dtype: float64, Exposure               10300
Eligible Collateral      300
Addon                    100
RW                         1
dtype: int64)The analysis of the components for the top 2 netting sets with the most change in 'RWA' reveals the following:

1. **Netting Set ID 101**:
   - The 'Exposure' increased significantly from 20,021 to 100,010,021, which is the primary driver of the large change in 'RWA'.
   - The 'Eligible Collateral', 'Addon', and 'RW' remained the same, so they did not contribute to the change.

2. **Netting Set ID 102**:
   - The 'RW' changed from 0.02 to 1, which is the primary driver of the change in 'RWA'.
   - The 'Exposure', 'Eligible Collateral', and 'Addon' remained the same, so they did not contribute to the change.

Final Answer: The top 2 netting sets with the most change in 'RWA' are:
1. Netting Set ID 101: The change is driven by a significant increase in 'Exposure'.
2. Netting Set ID 102: The change is driven by an increase in 'RW' from 0.02 to 1.

> Finished chain.
{'input': "\nIn both df_current and df_updated, the column 'EAD' is calculated as Exposure + Eligible Collateral + Addon, the column 'RWA' is calculated as EAD * RW.\nUsing df_current and df_updated, identify the top 2 netting set with the most change in 'RWA'and provide the driver of the change. \n", 
'output': "The top 2 netting sets with the most change in 'RWA' are:
1. Netting Set ID 101: The change is driven by a significant increase in 'Exposure'.
2. Netting Set ID 102: The change is driven by an increase in 'RW' from 0.02 to 1."}

"""