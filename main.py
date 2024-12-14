from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from globals import CLASSIFICATION, REGRESSION
from ml.db.db import database
from ml.server import predict_router

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await database.connect()
    REGRESSION.load_model("regression")
    CLASSIFICATION.load_model("classification")


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


app.include_router(predict_router.router, prefix="/ml", tags=["ml"])


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}
