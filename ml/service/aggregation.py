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

    async def aggregate(self):
        query = """
                with cars_aggregate as (
                select 
                      car_id,
                      AVG(rating) AS avg_rating,
                      AVG(ride_duration) AS avg_ride_duration,
                      MIN(ride_duration) as min_ride_duration,
                      MAX(ride_duration) as max_ride_duration,
                      AVG(ride_cost) as avg_ride_cost,
                      AVG(speed_avg) as avg_speed,
                      AVG(speed_max) as avg_speed_max,
                      SUM(stop_times) as sum_stop_times,
                      SUM(distance) as total_distance,
                      SUM(refueling) AS total_refueling,
                      AVG(driver_rating) as avg_user_rating,
                      COUNT(driver_time_accidents) as accidents
                from
                    ride_info
                join rides using (ride_id)
                join drivers using (driver_id)
                group by car_id)


                select
                     model,
                     car_type,
                     fuel_type,
                     car_rating,
                     year_to_start,
                     rides as riders,
                     year_to_work,
                     avg_rating,
                     avg_ride_duration,
                     min_ride_duration,
                     max_ride_duration,
                     avg_ride_cost,
                     avg_speed,
                     avg_speed_max,
                     sum_stop_times,
                     total_distance,
                     total_refueling,
                     avg_user_rating,
                     accidents,
                     target_reg,
                     target_class,
                     car_id
                from cars
                join cars_aggregate using(car_id)
                join cars_predicted_data using(car_id)
        """

        try:
            rows = await self.db.fetch_all(query)
            train = [dict(record._row) for record in rows]
            self.data = pd.DataFrame(train)
            logging.info("Агрегация данных выполнена успешно.")

        except Exception as error:
            logging.error(f"Ошибка при агрегации данных: {error}")

    async def train_aggregate(self):
        query = """
                with cars_aggregate as (
        select 
              car_id,
              AVG(rating) AS avg_rating,
              AVG(ride_duration) AS avg_ride_duration,
              MIN(ride_duration) as min_ride_duration,
              MAX(ride_duration) as max_ride_duration,
              AVG(ride_cost) as avg_ride_cost,
              AVG(speed_avg) as avg_speed,
              AVG(speed_max) as avg_speed_max,
              SUM(stop_times) as sum_stop_times,
              SUM(distance) as total_distance,
              SUM(refueling) AS total_refueling,
              AVG(driver_rating) as avg_user_rating,
              COUNT(driver_time_accidents) as accidents
        from
            ride_info
        join rides using (ride_id)
        join drivers using (driver_id)
        group by car_id)


        select
             model,
             car_type,
             fuel_type,
             car_rating,
             year_to_start,
             rides as riders,
             year_to_work,
             avg_rating,
             avg_ride_duration,
             min_ride_duration,
             max_ride_duration,
             avg_ride_cost,
             avg_speed,
             avg_speed_max,
             sum_stop_times,
             total_distance,
             total_refueling,
             avg_user_rating,
             accidents,
             car_id
        from cars
        join cars_aggregate using(car_id)
        join cars_predicted_data using(car_id)
        """

        try:
            rows = await self.db.fetch_all(query)
            train = [dict(record._row) for record in rows]
            self.data = pd.DataFrame(train)
            logging.info("Агрегация данных выполнена успешно.")

        except Exception as error:
            logging.error(f"Ошибка при агрегации данных: {error}")

    def get_data(self) -> pd.DataFrame:
        if self.data is not None:
            return self.data
        else:
            logging.warning("Данные не загружены.")
            return pd.DataFrame()
