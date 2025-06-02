from unittest import TextTestRunner
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import dspy
import logging
from dspy.utils.logging_utils import DSPyLoggingStream
import sys
import os
from dotenv import load_dotenv

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["mcp_server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)


class DSPyAirlineCustomerService(dspy.Signature):
    """You are an airline customer service agent. You are given a list of tools to handle user requests.
    You should decide the right tool to use in order to fulfill users' requests."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Message that summarizes the process result, and the information users need, "
            "e.g., the confirmation_number if it's a flight booking request."
        )
    )


dspy.enable_logging()
dspy.enable_litellm_logging()


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger("dspy").setLevel(logging.DEBUG)

root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

LLM_URL=os.getenv('LLM_URL')
API_KEY=os.getenv('API_KEY')
LLM_MODEL=os.getenv('LLM_MODEL')
dspy.enable_logging()
lm = dspy.LM(LLM_MODEL,
             api_base=LLM_URL,  # ensure this points to your port
             api_key=API_KEY, model_type='chat')


LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"
dspy.configure(lm=lm)
# logging.config.dictConfig(
#     {
#         "version": 1,
#         "disable_existing_loggers": False,
#         "formatters": {
#             "dspy_formatter": {
#                 "format": LOGGING_LINE_FORMAT,
#                 "datefmt": LOGGING_DATETIME_FORMAT,
#             },
#         },
#         "handlers": {
#             "dspy_handler": {
#                 "formatter": "dspy_formatter",
#                 "class": "logging.StreamHandler",
#                 "stream": DSPyLoggingStream(),
#             },
#         },
#         "loggers": {
#             "dspy": {
#                 "handlers": ["dspy_handler"],
#                 "level": "DEBUG",
#                 "propagate": TextTestRunner,
#             },
#         },
#     }
# )


async def run(user_request):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            tools = await session.list_tools()

            # Convert MCP tools to DSPy tools
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            # Create the agent
            react = dspy.ReAct(DSPyAirlineCustomerService, tools=dspy_tools)

            result = await react.acall(user_request=user_request)
            print(result)
            dspy.inspect_history(n=50)
            dspy


if __name__ == "__main__":
    import asyncio

    asyncio.run(run("please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam"))