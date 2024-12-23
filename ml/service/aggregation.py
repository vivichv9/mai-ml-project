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
            WITH subq AS (
	select 
		car_id,
		AVG(rating) as avg_rating,
		AVG(ride_duration) as avg_ride_duration,
		MIN(ride_duration) as min_ride_duration,
		MAX(ride_duration) as max_ride_duration,
		AVG(ride_cost) as avg_ride_cost,
		AVG(speed_avg) as avg_speed,
		AVG(speed_max) as avg_speed_max,
		SUM(stop_times) as sum_stop_times,
		SUM(distance) as total_distance,
		SUM(refueling) as total_refueling,
		AVG(user_rating) as avg_user_rating,
		AVG(user_time_accident) as accidents
	from 
		rides_info ri join driver_info di USING(user_id)
	group by car_id
)

select * from car_train join subq USING(car_id)
            """

        try:
            rows = await self.db.fetch_all(query)
            train = [dict(record._row) for record in rows]
            self.data = pd.DataFrame(train)
            logging.info("Агрегация данных выполнена успешно.")

        except Exception as error:
            logging.error(f"Ошибка при агрегации данных: {error}") 

    async def aggregate_data_test(self):
        query = """
        WITH subq AS (
            select 
                car_id,
                AVG(rating) as avg_rating,
                AVG(ride_duration) as avg_ride_duration,
                MIN(ride_duration) as min_ride_duration,
                MAX(ride_duration) as max_ride_duration,
                AVG(ride_cost) as avg_ride_cost,
                AVG(speed_avg) as avg_speed,
                AVG(speed_max) as avg_speed_max,
                SUM(stop_times) as sum_stop_times,
                SUM(distance) as total_distance,
                SUM(refueling) as total_refueling,
                AVG(user_rating) as avg_user_rating,
                AVG(user_time_accident) as accidents
            from 
                rides_info ri join driver_info di USING(user_id)
            group by car_id
        )

        select * from car_test join subq USING(car_id)
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
