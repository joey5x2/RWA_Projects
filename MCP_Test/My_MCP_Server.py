from fastmcp import FastMCP
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any

# Instantiate FastMCP server
mcp = FastMCP("MyServer")

# Define a Pydantic model for the DataFrame output (can be simplified to Dict[str, Any] for basic cases)
class DataFrameOutput(BaseModel):
    data: Dict[str, Any] # Dictionary representation of the DataFrame

# Define a tool using the @mcp.tool decorator
@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds two numbers and returns the sum."""
    return a + b

# @mcp.tool()
# def load_file() -> DataFrameOutput:
#     """
#     Reads a CSV file from the given path and returns its content as a Pandas DataFrame.
#     """

#     file_path = "C:\\Users\\joexz\\Projects\\MCP\\Data1 v3.csv"
#     try:
#         df = pd.read_csv(file_path)
#         # Convert DataFrame to a dictionary for serialization
#         return DataFrameOutput(data=df.to_dict(orient='records'))
#     except FileNotFoundError:
#         raise ValueError(f"File not found at: {file_path}")
#     except Exception as e:
#         raise ValueError(f"Error reading CSV file: {e}")

@mcp.tool()
async def get_csv_data(csv_file_path: str) -> str: #pd.DataFrame:
    """
    Reads a pre-defined CSV file and returns its content as a Pandas DataFrame.
    """
    # In a real application, you might load this path from an environment variable
    # or a configuration file. The key is that the client doesn't provide it.
    
    # csv_file_path = "Data1v3.csv" 
    try:
        df = pd.read_csv(csv_file_path)
        data = df.to_json()
        return data
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        # You might raise a more specific FastMCP error here for client handling
        raise

# Run the server using stdio transport (for local testing)
if __name__ == "__main__":
    mcp.run(transport="stdio")
