from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class Stats(BaseModel):
    id: str

@router.get("/", response_model=Stats)
def read_stats() -> Any:
    """
    Retrieve string
    """
    return Stats(id="foobar")
