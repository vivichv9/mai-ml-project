import pandas as pd
from databases import Database
from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException

from ml.db.db import get_db
from model_env import CLASSIFICATION, REGRESSION

router = APIRouter()


@router.get("/predict")
async def prediction(db: Database = Depends(get_db)):
    # TODO select data from database

    train = await db.fetch_all("SELECT * FROM public.car_train")

    if train is None:
        raise HTTPException(
            status_code=500, detail="Server Error: Database response is empty!"
        )

    train_data = [dict(record._row) for record in train]
    train_df = pd.DataFrame(
        train_data,
        columns=[
            "car_id",
            "model",
            "car_type",
            "fuel_type",
            "car_rating",
            "year_to_start",
            "riders",
            "year_to_work",
            "target_reg",
            "target_class",
        ],
    )

    train_x = train_df.drop(["target_reg", "target_class"], axis=1)
    predicted_class = CLASSIFICATION.prediction(train_x)
    predicted_value = REGRESSION.predicton(train_x)

    # TODO insert predicted values to database

    return {"predictions amount": predicted_class.shape[0]}
