import dspy
from pydantic import BaseModel, Field,TypeAdapter
import pathlib


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
    
drugs_json = pathlib.Path('fake-drug-data.json').read_text()
patients_json = pathlib.Path('fake-patient-data.json').read_text()
    
drugs = TypeAdapter(list[Drug]).validate_json(drugs_json)
print(len(drugs))

patients = TypeAdapter(list[Patient]).validate_json(patients_json)
print(len(patients))
    
