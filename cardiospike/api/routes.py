from fastapi import APIRouter, HTTPException

from cardiospike import SMART_MODEL_PATH
from cardiospike.api.inference import SmartModel
from cardiospike.api.models import RR, Model500, Predictions

router = APIRouter()

model = SmartModel(str(SMART_MODEL_PATH))

#
# @app.get("/")
# def index():
#     return "Visit /docs"


@router.post("/predict", responses={200: {"model": Predictions}, 500: {"model": Model500}})
def predict(rr: RR):
    try:
        anomaly_proba, anomaly_thresh, errors, error_thresh = model.predict(rr.sequence)

        return Predictions(
            study=rr.study,
            anomaly_proba=anomaly_proba,
            errors=errors,
            anomaly_thresh=anomaly_thresh,
            error_thresh=error_thresh,
        )
    except Exception as e:  # noqa
        raise HTTPException(status_code=500, detail=f"Something went wrong! Error:\n{e}")
