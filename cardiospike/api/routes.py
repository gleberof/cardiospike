from fastapi import APIRouter, HTTPException

from cardiospike.api.models import RR, Model500, Predictions
from cardiospike.inference import DummyModel

router = APIRouter()

model = DummyModel()

#
# @app.get("/")
# def index():
#     return "Visit /docs"


@router.post("/predict", responses={200: {"model": Predictions}, 500: {"model": Model500}})
def predict(rr: RR):
    try:
        predictions, errors = model.predict(rr.sequence)

        return Predictions(study=rr.study, predictions=predictions, errors=errors)
    except:  # noqa
        raise HTTPException(status_code=500, detail="Something went wrong!")
