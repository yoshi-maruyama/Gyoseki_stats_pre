from pydantic import BaseModel, validator
from typing import List, Union

class SummaryResponse(BaseModel):
    data: str

class SummaryRequest(BaseModel):
    returns: List[Union[float, int]]
    benchmark: List[Union[float, int]]
    start_date: str

    # returnsとbenchmarkの長さが等しいことを確認
    @validator('benchmark')
    def check_equal_length(cls, v, values):
        returns = values.get('returns')
        if returns is not None and len(v) != len(returns):
            raise ValueError('The number of elements in returns and benchmark must be equal')
        return v
