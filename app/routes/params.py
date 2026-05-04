from fastapi import APIRouter
from app.models.schemas import ControlParams

router = APIRouter()

params = ControlParams()

@router.get("/api/v1/params")
async def get_params() -> ControlParams:
    return params

@router.put("/api/v1/params")
async def update_params(new_params: ControlParams) -> ControlParams:
    global params
    params = new_params
    return params