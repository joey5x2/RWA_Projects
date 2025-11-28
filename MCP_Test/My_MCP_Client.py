import asyncio
from fastmcp import Client
import pandas as pd

async def main():
    # connect to a server running locally
    # server_config = {"command":"python", "args":["My_MCP_Server"]}

    # Connect to the server via stdio (assuming the server script is named 'My_MCP_Server.py')
    # async with Client(server_config) as client:
    async with Client("My_MCP_Server.py") as client:
        # List available tools on the server
        tools = await client.list_tools()
        print(f"Available tools: {tools}")

        # Call the 'add' tool with arguments
        result = await client.call_tool("add", {"a": 10, "b": 5})
        print(f"Result of addition: {result.content[0].text}")

        # # Load csv file
        # data_path = "Data1 v3.csv"
        # result = await client.call_tool("load_file", {"a": 10})
        
        # # # Reconstruct DataFrame from the received dictionary
        # # df_received = pd.DataFrame(result["data"])
        
        # # print("DataFrame received from server:")
        # # print(df_received)

        try:
            # Call the get_csv_data tool on the server
            # result = await client.get_csv_data()
            file_path = "Data1v3.csv" 
            result = await client.call_tool("get_csv_data", {"csv_file_path":file_path})

            print(type(result.content[0].text))
            # print(result.content[0].text)
            data = pd.read_json(result.content[0].text)
            # print(data)
            
            if isinstance(data, pd.DataFrame):
                print("Received DataFrame from server:")
                print(data)
            else:
                print("Received unexpected data type from server:", type(result))

        except Exception as e:
            print(f"Error calling get_csv_data: {e}")

if __name__ == "__main__":
    asyncio.run(main())
