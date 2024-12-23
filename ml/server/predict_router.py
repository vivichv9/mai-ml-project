import pandas as pd
from databases import Database
from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from ml.service.aggregation import Aggregation
from ml.db.db import get_db
from model_env import CLASSIFICATION, REGRESSION
from catboost import Pool

router = APIRouter()


@router.get("/predict")
async def prediction(db: Database = Depends(get_db)):
    agg = Aggregation()
    await agg.aggregate_data_test()

    data = agg.get_data()
    data.drop(['car_id'], axis=1, inplace=True)
    cat_features = ["model", "car_type", "fuel_type"]
    data = Pool(data=data, cat_features=cat_features)

    predicted_class = CLASSIFICATION.prediction(data)
    predicted_value = REGRESSION.prediction(data)

    predicted_class = predicted_class.tolist() if hasattr(predicted_class, "tolist") else predicted_class
    predicted_value = predicted_value.tolist() if hasattr(predicted_value, "tolist") else predicted_value

    predictions = [
        {"predicted_class": cls, "predicted_value": value}
        for cls, value in zip(predicted_class, predicted_value)
    ]

    return {"predictions": predictions}