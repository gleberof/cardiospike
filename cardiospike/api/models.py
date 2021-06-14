from typing import List

from pydantic import BaseModel, Field

study_field = Field(title="study", description="unique rr study identifier", example="user_id_0")


class RR(BaseModel):
    study: str = study_field
    sequence: List[int] = Field(
        title="sequence", description="RR intervals in milliseconds", example=[200, 300, 200, 300, 200, 300]
    )


class Predictions(BaseModel):
    study: str = study_field
    predictions: List[int] = Field(
        title="predictions", description="list of anomaly predictions (0 or 1)", example=[0, 0, 1, 1, 1, 0]
    )
    errors: List[int] = Field(title="errors", description="list errors (0 or 1)", example=[1, 0, 0, 0, 0, 0])


class Model500(BaseModel):
    message: str
