from typing import List

from pydantic import BaseModel, Field

study_field = Field(title="study", description="unique rr study identifier", example="user_id_0")


class RR(BaseModel):
    study: str = study_field
    sequence: List[int] = Field(
        title="Sequence", description="RR intervals in milliseconds", example=[200, 300, 200, 300, 200, 300]
    )


class Predictions(BaseModel):
    study: str = study_field
    anomaly_proba: List[float] = Field(
        title="Anomaly Probabilities",
        description="list of anomaly prediction probabilities",
        example=[0.5, 0.2, 1.0, 0.2, 0.8, 0.1],
    )
    anomaly_thresh: float = Field(
        title="Anomaly Threshold",
        description="threshold for anomaly probabilities separation to `detected` and `not detected`",
        example=0.4,
    )
    errors: List[int] = Field(
        title="Observation Error",
        description="list of observation errors predicted by the model (0 or 1)",
        example=[0, 1, 0, 1, 1, 0],
    )
    error_thresh: float = Field(
        title="Error Threshold",
        description="threshold for error probabilities separation to `detected` and `not detected` ",
        example=0.4,
    )


class Model500(BaseModel):
    message: str
