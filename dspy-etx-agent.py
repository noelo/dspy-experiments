import dspy
import dspy.predict
from pydantic import BaseModel, Field,TypeAdapter
from typing_extensions import TypedDict
import pathlib
import logging
from dspy.utils.logging_utils import DSPyLoggingStream
import sys
import os
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient

load_dotenv()
LLAMA_STACK_URL = os.getenv("LLAMA_STACK_URL")


root = logging.getLogger()
root.setLevel(logging.WARN)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

class GithubIssue(TypedDict):
    repo: str
    title: str 
    body: str  
    
class ExampleDataSet(TypedDict):
    logs: str
    gh_cmd: str 

class GithubOperation(BaseModel):
    name: str = Field(alias="name",default="create_issue")
    arguments: GithubIssue
    
class OpenshiftIssueService(dspy.Signature):
    """You are an expert OpenShift administrator. Your will be provided logs and you need to analyze the logs, summarize the error, and create a GitHub issue."""

    log_message: str = dspy.InputField(desc=("contains the pod logs"))
    repo_name: str = dspy.InputField(desc=("Name of the github repo"))
    github: GithubOperation = dspy.OutputField(desc=("A github operation with the title of Pipeline Issue and body containing the log summerization"),prefix=" ")   
    
    
def static_run():
    pod_logs = pathlib.Path('pod-logs.txt').read_text()
    
    print(OpenshiftIssueService)
    pod_processor = dspy.Predict(OpenshiftIssueService)
    result= pod_processor(log_message=pod_logs,repo_name="Red Hat ETX Repo")
    print(result)
    dspy.inspect_history(n=50)
    # dspy   
    
    
def build_examples():     
    example_json = pathlib.Path('pod-log-training-examples.json').read_text()   
    training_examples = TypeAdapter(list[ExampleDataSet]).validate_json(example_json)
    
    print(len(training_examples))



if __name__ == "__main__":
    ex = build_examples()
    print(ex)
    lls_client = LlamaStackClient(base_url=LLAMA_STACK_URL)
    model_list = lls_client.models.list()
    llm = dspy.LM(
        "openai/" + model_list[0].identifier,
        api_base=LLAMA_STACK_URL + "/v1/openai/v1",
        model_type="chat",
        api_key="this is a fake key"
    )

    LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"
    dspy.configure(lm=llm)
    static_run()    
    
    
    
    
    
    
    
    

        
        # # Use .format() to avoid f-string curly brace conflicts
        # prompt_template = """You are an expert OpenShift administrator. Your task is to analyze pod logs, summarize the error, and generate a JSON object to create a GitHub issue for tracking. Follow the format in the examples below.
        
        # ---
        # EXAMPLE 1:
        # Input: The logs for pod 'frontend-v2-abcde' in namespace 'webapp' show: ImagePullBackOff: Back-off pulling image 'my-registry/frontend:latest'.

        # Output:
        # The pod is in an **ImagePullBackOff** state. This means Kubernetes could not pull the container image 'my-registry/frontend:latest', likely due to an incorrect image tag or authentication issues.
        # {"name":"create_issue","arguments":{"owner":"redhat-ai-services","repo":"etx-agentic-ai","title":"Issue with Etx pipeline","body":"### Cluster/namespace location\\nwebapp/frontend-v2-abcde\\n\\n### Summary of the problem\\nThe pod is failing to start due to an ImagePullBackOff error.\\n\\n### Detailed error/code\\nImagePullBackOff: Back-off pulling image 'my-registry/frontend:latest'\\n\\n### Possible solutions\\n1. Verify the image tag 'latest' exists in the 'my-registry/frontend' repository.\\n2. Check for authentication errors with the image registry."}}

        # ---
        # EXAMPLE 2:
        # Input: The logs for pod 'data-processor-xyz' in namespace 'pipelines' show: CrashLoopBackOff. Last state: OOMKilled.

        # Output:
        # The pod is in a **CrashLoopBackOff** state because it was **OOMKilled**. The container tried to use more memory than its configured limit.
        # {"name":"create_issue","arguments":{"owner":"redhat-ai-services","repo":"etx-agentic-ai","title":"Issue with Etx pipeline","body":"### Cluster/namespace location\\npipelines/data-processor-xyz\\n\\n### Summary of the problem\\nThe pod is in a CrashLoopBackOff state because it was OOMKilled (Out of Memory).\\n\\n### Detailed error/code\\nCrashLoopBackOff, Last state: OOMKilled\\n\\n### Possible solutions\\n1. Increase the memory limit in the pod's deployment configuration.\\n2. Analyze the application for memory leaks."}}
        # ---

        # NOW, YOUR TURN:

        # Input: Review the OpenShift logs for the pod '{pod_name}' in the '{namespace}' namespace. If the logs indicate an error, search for the solution, create a summary message with the category and explanation of the error, and create a Github issue using {{"name":"create_issue","arguments":{{"owner":"redhat-ai-services","repo":"etx-agentic-ai","title":"Issue with Etx pipeline","body":"<summary of the error>"}}}}. DO NOT add any optional parameters.

        # ONLY tail the last 10 lines of the pod, no more.
        # The JSON object formatted EXACTLY as outlined above.
        # """