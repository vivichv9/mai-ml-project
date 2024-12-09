import logging
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2 import Error


class Aggregation:
    def __init__(self, env_path="../.env"):
        load_dotenv(env_path)
        self.PG_USER: str = os.getenv("PG_USER")
        self.PG_PASSWORD: str = os.getenv("PG_PASSWORD")
        self.PG_HOST: str = os.getenv("PG_HOST")
        self.PG_PORT: str = os.getenv("PG_PORT")
        self.PG_DATABASE: str = os.getenv("PG_DATABASE")
        self.data: pd.DataFrame = None

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def aggregate_data(self):
        query = """
        SELECT
            d.age,
            d.user_rating,
            d.user_rides,
            d.user_time_accident,
            d.sex,
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
            AVG(r.deviation_normal) AS average_deviation_normal,
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
            driver_info d
        LEFT JOIN
            rides_info r
        ON
            d.user_id = r.user_id
        LEFT JOIN
            car_train ct
        ON
            r.car_id = ct.car_id
        WHERE
            r.ride_duration < 300
            AND r.distance < 7000
            AND r.ride_cost <= 4000
        GROUP BY
            d.age,
            d.user_rating,
            d.user_rides,
            d.user_time_accident,
            d.sex,
            ct.model,
            ct.car_type,
            ct.fuel_type,
            ct.car_rating,
            ct.year_to_start,
            ct.riders,
            ct.year_to_work,
            ct.target_reg,
            ct.target_class;
        """

        try:
            connection = psycopg2.connect(
                user=self.PG_USER,
                password=self.PG_PASSWORD,
                host=self.PG_HOST,
                port=self.PG_PORT,
                database=self.PG_DATABASE,
            )
            logging.info("The connection to the database was successfully established.")

            self.data = pd.read_sql_query(query, connection)
            logging.info("Data aggregation completed successfully.")

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
            logging.info("Missing values ​​are filled in.")

        except (Exception, Error) as error:
            logging.error(f"PostgreSQL Error: {error}")
        finally:
            if "connection" in locals() and connection:
                connection.close()
                logging.info("PostgreSQL connection is closed.")

    def get_data(self) -> pd.DataFrame:
        if self.data is not None:
            return self.data
        else:
            logging.warning("The data was not loaded.")
            return pd.DataFrame()
