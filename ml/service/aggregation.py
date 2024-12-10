import logging

import pandas as pd


class Aggregation:
    def __init__(self):
        self.data: pd.DataFrame = None

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    async def aggregate_data(self):
        query = """
        WITH ride_aggregates AS (
        SELECT
            r.car_id,
            COUNT(r.ride_id) AS total_rides,
            AVG(r.rating) AS average_rating,
            MIN(r.rating) AS min_rating,
            SUM(r.ride_duration) AS total_ride_duration,
            AVG(r.ride_duration) AS average_ride_duration,
            SUM(r.ride_cost) AS total_ride_cost,
            AVG(r.speed_avg) AS average_speed,
            MIN(r.speed_max) AS min_speed,
            AVG(r.speed_max) AS average_max_speed,
            MAX(r.speed_max) AS max_speed,
            SUM(r.distance) AS total_distance,
            SUM(r.refueling) AS total_refuelings,
            AVG(r.user_ride_quality) AS average_ride_quality,
            MIN(r.user_ride_quality) AS min_ride_quality,
            AVG(r.deviation_normal) AS average_deviation_normal
        FROM
            rides_info r
        JOIN
            driver_info d ON d.user_id = r.user_id
        WHERE
            r.ride_duration < 300
            AND r.distance < 7000
            AND r.ride_cost <= 4000
        GROUP BY
            r.car_id
        ),
        car_details AS (
            SELECT
                ct.car_id,
                ct.model,
                ct.car_type,
                ct.fuel_type,
                ct.car_rating,
                ct.year_to_start,
                ct.riders,
                ct.year_to_work,
                ct.target_reg,
                ct.target_class
            FROM
                car_train ct
        )

            SELECT
                cd.car_id,
                cd.model,
                cd.car_type,
                cd.fuel_type,
                cd.car_rating,
                cd.year_to_start,
                cd.riders,
                cd.year_to_work,
                cd.target_reg,
                cd.target_class,
                COALESCE(ra.total_rides, 0) AS total_rides,
                COALESCE(ra.average_rating, 0) AS average_rating,
                COALESCE(ra.min_rating, 0) AS min_rating,
                COALESCE(ra.total_ride_duration, 0) AS total_ride_duration,
                COALESCE(ra.average_ride_duration, 0) AS average_ride_duration,
                COALESCE(ra.total_ride_cost, 0) AS total_ride_cost,
                COALESCE(ra.average_speed, 0) AS average_speed,
                COALESCE(ra.min_speed, 0) AS min_speed,
                COALESCE(ra.average_max_speed, 0) AS average_max_speed,
                COALESCE(ra.max_speed, 0) AS max_speed,
                COALESCE(ra.total_distance, 0) AS total_distance,
                COALESCE(ra.total_refuelings, 0) AS total_refuelings,
                COALESCE(ra.average_ride_quality, 0) AS average_ride_quality,
                COALESCE(ra.min_ride_quality, 0) AS min_ride_quality,
                COALESCE(ra.average_deviation_normal, 0) AS average_deviation_normal
            FROM
                car_details cd
            LEFT JOIN
                ride_aggregates ra ON cd.car_id = ra.car_id;
            """

        try:
            rows = await db.fetch_all(query)
            self.data = pd.DataFrame(rows)
            logging.info("Агрегация данных выполнена успешно.")

            self.data.fillna(
                {
                    "total_rides": 0,
                    "average_rating": 0,
                    "min_rating": 0,
                    "total_ride_duration": 0,
                    "average_ride_duration": 0,
                    "total_ride_cost": 0,
                    "average_speed": 0,
                    "min_speed": 0,
                    "average_max_speed": 0,
                    "max_speed": 0,
                    "total_distance": 0,
                    "total_refuelings": 0,
                    "average_ride_quality": 0,
                    "min_ride_quality": 0,
                    "average_deviation_normal": 0,
                    "model": "Unknown",
                    "car_type": "Unknown",
                    "fuel_type": "Unknown",
                    "car_rating": 0.0,
                    "year_to_start": 0,
                    "riders": 0,
                    "year_to_work": 0,
                    "target_reg": 0.0,
                    "target_class": "Unknown",
                },
                inplace=True,
            )
            logging.info("Отсутствующие значения заполнены.")

        except Exception as error:
            logging.error(f"Ошибка при агрегации данных: {error}")

    def get_data(self) -> pd.DataFrame:
        if self.data is not None:
            return self.data
        else:
            logging.warning("Данные не загружены.")
            return pd.DataFrame()
