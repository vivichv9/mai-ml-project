import pandas as pd
from catboost import Pool
from databases import Database
from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException

from config import CLASSIFICATION, REGRESSION
from ml.db.db import get_db
from ml.service.aggregation import Aggregation, logging

router = APIRouter()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

async def upsert_car_predictions(
    data,
    db: Database
):

    if not data:
        logging.warning("Нет данных.")
        return


    query = """
        INSERT INTO cars_predicted_data (car_id, target_reg, target_class)
        VALUES (:car_id, :target_reg, :target_class)
        ON CONFLICT (car_id) DO UPDATE
        SET
            target_reg = EXCLUDED.target_reg,
            target_class = EXCLUDED.target_class;
    """

    try:
        await db.execute_many(query=query, values=data)
        logging.info(f"Upsert  выполнен успешно для {len(data)} записей.")
    except Exception as e:
        logging.error(f"Ошибка при выполнении upsert: {e}")


@router.get("/predict")
async def prediction(db: Database = Depends(get_db)):
    agg = Aggregation()
    await agg.train_aggregate()

    data = agg.get_data()
    car_ids = data['car_id'].tolist()
    data.drop(["car_id"], axis=1, inplace=True)
    cat_features = ["model", "car_type", "fuel_type"]
    data = Pool(data=data, cat_features=cat_features)

    predicted_class = CLASSIFICATION.prediction(data)
    predicted_value = REGRESSION.prediction(data)

    predicted_class = (
        predicted_class.tolist()
        if hasattr(predicted_class, "tolist")
        else predicted_class
    )

    predicted_value = (
        predicted_value.tolist()
        if hasattr(predicted_value, "tolist")
        else predicted_value
    )

    predicted_class_flat = []
    for idx, cls in enumerate(predicted_class):
        predicted_class_flat.append(cls[0])

    predictions = [
        {"predicted_class": cls[0], "predicted_value": value}
        for cls, value in zip(predicted_class, predicted_value)
    ]

    car_predictions = []
    for cid, cls, val in zip(car_ids, predicted_class_flat, predicted_value):
        car_pred = {
            "car_id": cid,
            "target_reg": val,
            "target_class": cls
        }
        car_predictions.append(car_pred)


    try:
        await upsert_car_predictions(car_predictions, db)
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Ошибка при upsert: {e}")


    return {"predictions": predictions}
