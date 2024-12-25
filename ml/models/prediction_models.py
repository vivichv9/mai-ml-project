from pydantic import BaseModel


class PredictionRequest(BaseModel):
    amount: int


class PredictionReaponse(BaseModel):
    amount: int
    new_orders_amount: int
    old_orders_rewrite: int
