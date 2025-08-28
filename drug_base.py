import dspy
import dspy.predict
from pydantic import BaseModel, Field,TypeAdapter
import pathlib
from unittest import TextTestRunner
import logging
from dspy.utils.logging_utils import DSPyLoggingStream
import sys
import os
from dotenv import load_dotenv
from fastmcp import Client
import asyncio
from llama_stack_client import LlamaStackClient
from random import randrange

load_dotenv()
LLAMA_STACK_URL = os.getenv("LLAMA_STACK_URL")


root = logging.getLogger()
root.setLevel(logging.WARN)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)


class Drug(BaseModel):
    drug_name: str = Field(title="drug_name")
    drug_type: str = Field(alias="type_classification")
    condition: str = Field(alias="med_condition")
    drug_interactions: str = Field(title="drug_interactions")
    contra_indication:str = Field(alias="contra_indications")


class Patient(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    condition: str = Field(alias="primary_medical_condition")
    history: str = Field(alias="other_relevant_medical_history")
    notes: str 
    
class DrugAdvisorService(dspy.Signature):
    """You are a medical advisor assistant. You are given a list of drugs. You will be provided patient details and you need to determine what drug best addresses the patients condition taking their history and notes into account"""

    patient_details: Patient = dspy.InputField(desc=("contains the patients details as well as condition, medical history and medical notes"))
    drug_list: list[Drug] = dspy.InputField(desc=("contains the drug name, condition details , negative drug interactions and contra indications"))
    patient_age_group: str = dspy.OutputField(
        desc=(
            "What age group is the patient in?"
        )
    )
    drug_advice: str = dspy.OutputField(
        desc=(
            "The two most suitable drugs that addresses the patients condition but takes the patients medical condition, medical history and notes into account"
        )
    )
    drug_name: str = dspy.OutputField(
        desc=(
            "The name of the two most suitable drugs that addresses the patients condition but takes the patients medical condition, medical history and notes into account"
        )
    )
    
async def run():
    drugs_json = pathlib.Path('fake-drug-data.json').read_text()
    patients_json = pathlib.Path('fake-patient-data.json').read_text()
    drugs = TypeAdapter(list[Drug]).validate_json(drugs_json)
    patients = TypeAdapter(list[Patient]).validate_json(patients_json)

    react = dspy.predict(DrugAdvisorService)

    result = await react.acall(patient_details=patients[0],drug_list=drugs)
    print(result)
    dspy.inspect_history(n=50)
    dspy
    
def static_run():
    drugs_json = pathlib.Path('fake-drug-data.json').read_text()
    patients_json = pathlib.Path('fake-patient-data.json').read_text()
    drugs = TypeAdapter(list[Drug]).validate_json(drugs_json)
    patients = TypeAdapter(list[Patient]).validate_json(patients_json)
    print(f"Number of Patients : {len(patients)}")

    p_id= randrange(len(patients))
    # p_id=7
    
    print(f"\n Processing {patients[p_id]}")
    print(DrugAdvisorService)
    drug_predict = dspy.ChainOfThought(DrugAdvisorService)
    result= drug_predict(patient_details=patients[p_id],drug_list=drugs)
    print(result)
    dspy.inspect_history(n=50)
    dspy   


if __name__ == "__main__":
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