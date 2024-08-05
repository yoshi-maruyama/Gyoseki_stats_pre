from fastapi import FastAPI
from app.api.main import api_router
from mangum import Mangum

app = FastAPI()

# APIRouterをインクルード
app.include_router(api_router, prefix="/api")

# Mangumアダプターを利用してAWS Lambda対応
handler = Mangum(app)
