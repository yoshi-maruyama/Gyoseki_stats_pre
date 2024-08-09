from fastapi import FastAPI
from app.api.main import api_router
from app.middlewares.basicauth import BasicAuthMiddleware
from app.config import settings
from mangum import Mangum

app = FastAPI()
app.add_middleware(BasicAuthMiddleware)

# APIRouterをインクルード
app.include_router(api_router, prefix="/api")

@app.get("/")
def health_check():
    return {"message": "Hello from stats server"}

# Mangumアダプターを利用してAWS Lambda対応
handler = Mangum(app)
