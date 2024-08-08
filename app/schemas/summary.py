from pydantic import BaseModel

class SummaryResponse(BaseModel):
    data: str
