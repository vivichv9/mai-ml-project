from fastapi import APIRouter
from fastapi.exceptions import HTTPException

router = APIRouter()


@router.get("/predict")
async def prediction(req):
    raise HTTPException(status_code=500, detail="Not implemented!")
