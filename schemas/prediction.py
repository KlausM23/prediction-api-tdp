from pydantic import BaseModel
from typing import Optional

class Prediction(BaseModel):
    hash2 : Optional[float]
    hash : Optional[str]
    id : int
    year : int
    month : int
    hour : int
    dayOfTheWeek : int
    dayOfTheMonth : int
    latitude : Optional[float]
    longitude : Optional[float]