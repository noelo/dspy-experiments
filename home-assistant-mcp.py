import sys
import dspy
import logging
import os
import asyncio
from dspy.utils.logging_utils import DSPyLoggingStream

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from fastmcp.client.logging import LogMessage

load_dotenv()
LLM_URL=os.getenv('LLM_URL')
API_KEY=os.getenv('API_KEY')
LLM_MODEL=os.getenv('LLM_MODEL')
HA_KEY=os.getenv('HA_MCP_KEY')
HA_URL=os.getenv('HA_MCP_URL')

root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

dspy.enable_logging()


async def log_handler(message: LogMessage):
    level = message.level.upper()
    logger = message.logger or 'server'
    data = message.data
    print(f"[{level}] {logger}: {data}")
    
async def run(user_request):
    transport = SSETransport(
        url=HA_URL,
        auth=HA_KEY
    )
    
    # SSETransport add an addition slash at the end of the url which break HA
    transport.url=HA_URL
    client = Client(transport,log_handler=log_handler)
    async with client:
        tools = await client.list_tools()
        # Convert MCP tools to DSPy tools
        dspy_tools = []
        for tool in tools:
            print(f'{tool} \n')
            dspy_tools.append(dspy.Tool.from_mcp_tool(client.session, tool))

        # # Create the agent
        # react = dspy.ReAct(DSPyAirlineCustomerService, tools=dspy_tools)

        # result = await react.acall(user_request=user_request)
        # print(result)
        # dspy.inspect_history(n=50)
        # dspy


if __name__ == "__main__":
    asyncio.run(run("please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam"))