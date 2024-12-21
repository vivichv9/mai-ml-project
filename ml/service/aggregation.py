import logging

import pandas as pd

from ml.db.db import get_db


class Aggregation:
    def __init__(self):
        self.data: pd.DataFrame = None
        self.db = get_db()
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    async def aggregate_data(self):
        query = """
                    with cars_aggregate as (select
                    car_id,
                    COUNT(ride_id) AS total_rides,
                    AVG(rating) AS average_rating,
                    MIN(rating) AS min_rating,
                    SUM(ride_duration) AS total_ride_duration,
                    AVG(ride_duration) AS average_ride_duration,
                    SUM(ride_cost) AS total_ride_cost,
                    AVG(speed_avg) AS average_speed,
                    MIN(speed_max) AS min_speed,
                    AVG(speed_max) AS average_max_speed,
                    MAX(speed_max) AS max_speed,
                    SUM(distance) AS total_distance,
                    SUM(refueling) AS total_refuelings,
                    AVG(user_ride_quality) AS average_ride_quality,
                    MIN(user_ride_quality) AS min_ride_quality,
                    AVG(deviation_normal) AS average_deviation_normal
         from
           ride_info
         join rides using (ride_id)
         join drivers using (driver_id)
         where
           ride_duration < 300
           and distance < 7000
           and ride_cost <= 4000
         group by
           car_id)

        select *
        from cars
        join
            cars_aggregate using(car_id)
        join
            cars_predicted_data using(car_id)

            """

        try:
            rows = await self.db.fetch_all(query)
            self.data = pd.DataFrame(rows)
            logging.info("Агрегация данных выполнена успешно.")

        except Exception as error:
            logging.error(f"Ошибка при агрегации данных: {error}")

    def get_data(self) -> pd.DataFrame:
        if self.data is not None:
            return self.data
        else:
            logging.warning("Данные не загружены.")
            return pd.DataFrame()
