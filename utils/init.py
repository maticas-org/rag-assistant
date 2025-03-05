import os
import yaml

from typing             import Dict, List, Union
from os                 import getenv
from dotenv             import load_dotenv
from langchain_aws      import ChatBedrockConverse
from langchain_ollama   import ChatOllama   

# Load the environment variables and configs
load_dotenv()
with open('config-aws.yaml', 'r') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)
#print(f"Loaded configs:\n {configs}")

def check_env_vars():
    """
    Check if the required environment variables are set.
    """
    required_vars = ["AWS_ACCESS_KEY_ID",
                     "AWS_SECRET_ACCESS_KEY",
                     "AWS_REGION"]
    for var in required_vars:
        if var not in os.environ:
            raise Exception(f"Environment variable {var} is not set.")

def get_llm(usecase: str = "default") -> Union[ChatBedrockConverse, ChatOllama]:

    allowed_usecases = ["default", "semantic_grouping",
                        "summary", "extract_entity_types"]

    if usecase not in allowed_usecases:
        raise Exception(f"Invalid LLM usecase: {usecase}. Allowed values are: {allowed_usecases}")
    
    provider    = configs["backend"]["llm"][usecase]["provider"]
    model_name  = configs["backend"]["llm"][usecase]["model_name"]
    parameters  = configs["backend"]["llm"][usecase]["parameters"]

    # Initialize the AWS LLM Converse API
    if  provider == 'aws':
        check_env_vars()
        LLM = ChatBedrockConverse(
            model = model_name,
            **parameters
        )
    
    # Initialize the Ollama API
    elif provider == 'ollama':
        LLM = ChatOllama(
            model = model_name,
            **parameters
        )

    else:
        raise Exception("Invalid LLM provider in the configs.")

    return LLM