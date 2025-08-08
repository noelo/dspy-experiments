import sys
import dspy
import logging
import os
import asyncio

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.logging import LogMessage
from fastmcp.client.transports import StreamableHttpTransport
from llama_stack_client import LlamaStackClient


load_dotenv()
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
    lls_client = LlamaStackClient(base_url=LLAMA_STACK_URL)
    tool_group = lls_client.toolgroups.list()

    mcp_tools = [
        tool for tool in tool_group if tool.provider_id == "model-context-protocol"
    ]

    tool_list = {}
    for tool in mcp_tools:
        tool_list[tool.identifier] = tool.mcp_endpoint.uri
    return tool_list


async def convert_tools_dspy() -> list[dspy.adapters.types.tool.Tool]:
    tools_list = lls_get_tools()
    dspy_tools = []

    for tool in tools_list:
        print(tools_list[tool])
        mcp_client = Client(
            StreamableHttpTransport(url=tools_list[tool]), log_handler=log_handler
        )
        await mcp_client._connect()
        print(
            f" Connected:{mcp_client.is_connected()} Init result:{mcp_client.initialize_result}\n"
        )
        provided_tools = await mcp_client.list_tools()

        for method in provided_tools:
            dspy_tools.append(dspy.Tool.from_mcp_tool(mcp_client.session, method))
    return dspy_tools


async def dspy_mcp(user_request: str):
    dspy_tools = await convert_tools_dspy()
    react = dspy.ReAct(CustomerAssistantService, tools=dspy_tools)
    result = await react.acall(user_request=user_request)
    print(result)
    dspy.inspect_history(n=50)


def lls_get_inference_server():
    lls_client = LlamaStackClient(base_url=LLAMA_STACK_URL)
    model_list = lls_client.models.list()
    print(model_list)
    llm = dspy.LM(
        "openai/" + model_list[0].identifier,
        api_base=LLAMA_STACK_URL + "/v1/openai/v1",
        model_type="chat",
    )
    return llm


if __name__ == "__main__":
    dspy.enable_logging()
    llm = lls_get_inference_server()

    dspy.configure(lm=llm)
    asyncio.run(
        dspy_mcp(
            "help me find the recipe for cupcakes? I've already done a basic search and that didn't work. I also tried an advanced search and that didn't give great results"
        )
    )
