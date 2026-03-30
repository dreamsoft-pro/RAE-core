from pydantic import BaseModel
from typing import List

class ContractCompatibility(BaseModel):
    contract_name: str
    current_version: str
    compatible_with: List[str]
