"""Health-check endpoint used by load balancers and uptime monitors."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health", tags=["ops"])
async def health_check() -> JSONResponse:
    return JSONResponse({"status": "ok"})
