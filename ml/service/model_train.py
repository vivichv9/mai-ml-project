import asyncio

from ml.db.db import get_db
from ml.service.aggregation import Aggregation
from ml.service.prediction import BrokePredictor
from model_env import base_classification_model, base_regression_model


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
        ["target_class", "target_reg", "car_id", "car_number", "is_rented"],
        axis=1,
        inplace=True,
    )
    train_df = train_df.reset_index()
    train_df[["average_rating"]] = train_df[["average_rating"]].astype(float)
    train_df[["min_rating"]] = train_df[["min_rating"]].astype(float)
    train_df[["average_ride_duration"]] = train_df[["average_ride_duration"]].astype(
        float
    )
    train_df[["average_speed"]] = train_df[["average_speed"]].astype(float)
    train_df[["min_speed"]] = train_df[["min_speed"]].astype(float)
    train_df[["average_max_speed"]] = train_df[["average_max_speed"]].astype(float)
    train_df[["max_speed"]] = train_df[["max_speed"]].astype(float)
    train_df[["total_distance"]] = train_df[["total_distance"]].astype(float)
    train_df[["average_ride_quality"]] = train_df[["average_ride_quality"]].astype(
        float
    )
    train_df[["min_ride_quality"]] = train_df[["min_ride_quality"]].astype(float)
    train_df[["average_deviation_normal"]] = train_df[
        ["average_deviation_normal"]
    ].astype(float)
    train_df[["car_rating"]] = train_df[["car_rating"]].astype(float)

    classification_model = BrokePredictor(
        train_df, classification_target, base_classification_model
    )
    classification_model.param_selection()
    classification_model.train()
    classification_model.save_model("classification")

    regression_model = BrokePredictor(
        train_df, regression_target, base_regression_model
    )
    regression_model.param_selection()
    regression_model.train()
    regression_model.save_model("regression")


if __name__ == "__main__":
    asyncio.run(main())
