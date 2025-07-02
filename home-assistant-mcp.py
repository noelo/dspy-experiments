import sys
import dspy
import logging
import os
import asyncio

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
root.setLevel(logging.WARN)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

dspy.enable_logging()

lm = dspy.LM(LLM_MODEL,
             api_base=LLM_URL,  # ensure this points to your port
             api_key=API_KEY, model_type='chat')

dspy.configure(lm=lm)


class DSPyHomeAutomationService(dspy.Signature):
    """You are an home automation service agent. You are given a list of tools to handle user requests.
    You should decide the right tool to use in order to fulfill users' requests."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Message that summarizes the process result, and the information users need, "
            "e.g., the device has been turned on."
        )
    )

async def log_handler(message: LogMessage):
    level = message.level.upper()
    logger = message.logger or 'server'
    data = message.data
    print(f"[{level}] {logger}: {data}")
    
    
async def mcpInspect():
    transport = SSETransport(
        url=HA_URL,
        auth=HA_KEY
    )
    
    transport.url=HA_URL
    client = Client(transport,log_handler=log_handler)
    async with client:
        tools = await client.list_tools()
        # Convert MCP tools to DSPy tools
        for tool in tools:
            print(f'{tool} \n')
        prompts = await client.list_prompts()
        for prompt in prompts:
            print(f'{prompts} \n')
        resources = await client.list_resources()
        for resource in resources:
            print(f'{resource} \n')

            
async def executeCommand(user_request):
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

        # Create the agent
        react = dspy.ReAct(DSPyHomeAutomationService, tools=dspy_tools)

        result = await react.acall(user_request=user_request)
        print(result)
        dspy.inspect_history(n=50)
        dspy


if __name__ == "__main__":
    asyncio.run(executeCommand("check if the garage door is opened and if it is close it"))
    # asyncio.run(executeCommand("close the blind on the office western window"))
    # asyncio.run(mcpInspect())