import asyncio

from ml.db.db import get_db
from ml.service.aggregation import Aggregation
from ml.service.prediction import BrokePredictor
from catboost import CatBoostRegressor, CatBoostClassifier

async def main():
    db = get_db()
    await db.connect()
    agg = Aggregation()
    await agg.aggregate_data()
    await db.disconnect()

    train_df = agg.get_data()
    classification_target = train_df["target_class"]
    regression_target = train_df["target_reg"].astype(float)
    train_df.drop(
        ["target_class", "target_reg", "car_id"],
        axis=1,
        inplace=True,
    )
    train_df[["avg_rating"]] = train_df[["avg_rating"]].astype(float)
    train_df[["avg_ride_duration"]] = train_df[["avg_ride_duration"]].astype(float)
    train_df[["min_ride_duration"]] = train_df[["min_ride_duration"]].astype(
        float
    )
    train_df[["max_ride_duration"]] = train_df[["max_ride_duration"]].astype(float)
    train_df[["avg_ride_cost"]] = train_df[["avg_ride_cost"]].astype(float)
    train_df[["avg_speed"]] = train_df[["avg_speed"]].astype(float)
    train_df[["avg_speed_max"]] = train_df[["avg_speed_max"]].astype(float)
    train_df[["sum_stop_times"]] = train_df[["sum_stop_times"]].astype(float)
    train_df[["total_distance"]] = train_df[["total_distance"]].astype(
        float
    )
    train_df[["total_refueling"]] = train_df[["total_refueling"]].astype(float)
    train_df[["avg_user_rating"]] = train_df[
        ["avg_user_rating"]
    ].astype(float)
    train_df[["accidents"]] = train_df[["accidents"]].astype(float)

    classification_model = BrokePredictor(
        train_df, classification_target, CatBoostClassifier(
        loss_function="MultiClass",
        silent=False,
        )       
    )
    classification_model.train()
    classification_model.save_model("classification")

    regression_model = BrokePredictor(
        train_df, regression_target, CatBoostRegressor(loss_function="MAE", silent=False)
    )
    regression_model.train()
    regression_model.save_model("regression")


if __name__ == "__main__":
    asyncio.run(main())
