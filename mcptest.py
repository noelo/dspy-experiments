import sys
import dspy
import logging
import os
import asyncio

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from fastmcp.client.logging import LogMessage
from fastmcp.client.transports import StreamableHttpTransport
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger


load_dotenv()
LLM_URL = os.getenv("LLM_URL")
API_KEY = os.getenv("API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
LLAMA_STACK_URL = os.getenv("LLAMA_STACK_URL")


root = logging.getLogger()
root.setLevel(logging.WARN)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

class CustomerAssistantService(dspy.Signature):
    """You are a helpful customer service agent. You are given a list of tools to handle user requests.
    You should decide the right tool to use in order to fulfill users' requests."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Message that summarizes the process result, and the information users need, "
        )
    )

async def log_handler(message: LogMessage):
    level = message.level.upper()
    logger = message.logger or "server"
    data = message.data
    print(f"[{level}] {logger}: {data}")


def lls_get_tools():
    mcp_client = LlamaStackClient(base_url=LLAMA_STACK_URL)
    tool_group = mcp_client.toolgroups.list()

    mcp_tools = [
        tool for tool in tool_group if tool.provider_id == "model-context-protocol"
    ]

    tool_list = {}
    for tool in mcp_tools:
        # transport = StreamableHttpTransport(url=tool.mcp_endpoint.uri)
        # tool_list[tool.identifier] = Client(StreamableHttpTransport(url=tool.mcp_endpoint.uri), log_handler=log_handler)
        tool_list[tool.identifier] = tool.mcp_endpoint.uri   
    return tool_list


async def convert_tools_dspy()-> list[dspy.adapters.types.tool.Tool]:
    tools_list = lls_get_tools()    
    dspy_tools = []

    for tool in tools_list:
        print(tools_list[tool])
        mcp_client = Client(StreamableHttpTransport(url=tools_list[tool]),log_handler=log_handler)
        await mcp_client._connect()
        print(f" Connected:{mcp_client.is_connected()} Init result:{mcp_client.initialize_result}\n")
        provided_tools = await mcp_client.list_tools()
        
        for method in provided_tools:
            dspy_tools.append(dspy.Tool.from_mcp_tool(mcp_client.session, method))
            
    return dspy_tools
          
async def direct_mcp(user_request):    
    ls_client = LlamaStackClient(base_url=LLAMA_STACK_URL)
    tool_group = ls_client.toolgroups.list()

    mcp_tools = [
        tool for tool in tool_group if tool.provider_id == "model-context-protocol"
    ]
    transport = StreamableHttpTransport(
        url=mcp_tools[0].mcp_endpoint.uri
    )
    
    mcp_client = Client(transport,log_handler=log_handler)
    async with mcp_client:
        tools = await mcp_client.list_tools()
        dspy_tools = []
        for tool in tools:
            dspy_tools.append(dspy.Tool.from_mcp_tool(mcp_client.session, tool))

        # Create the agent
        react = dspy.ReAct(CustomerAssistantService, tools=dspy_tools)
        result = await react.acall(user_request=user_request)
        print(result)
        # dspy.inspect_history(n=50)
        # dspy
        
async def dspy_interact(user_request: str):
    dspy_tools= await(convert_tools_dspy())
    
    parsed_kwargs = dspy_tools[0]._validate_and_parse_args(user_request=user_request)
    print(parsed_kwargs)
    result = await dspy_tools[0].func(**parsed_kwargs)
    print(result)
    
async def dspy_mcp(user_request: str):
    dspy_tools= await(convert_tools_dspy())
    react = dspy.ReAct(CustomerAssistantService, tools=dspy_tools)
    result = await react.acall(user_request=user_request)
    print(result)
    dspy.inspect_history(n=50)
          


if __name__ == "__main__":
    dspy.enable_logging()
    lm = dspy.LM(
        LLM_MODEL,
        api_base=LLM_URL,  # ensure this points to your port
        api_key=API_KEY,
        model_type="chat",
    )
    dspy.configure(lm=lm)
    asyncio.run(dspy_mcp("help me find the recipe for cupcakes? I've already done a basic search and that didn't work"))
