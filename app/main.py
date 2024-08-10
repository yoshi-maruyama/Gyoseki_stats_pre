from fastapi import FastAPI
from app.api.main import api_router
from app.core.error_handler import create_exception_handler
from app.core.logger import setup_logger
from mangum import Mangum
from app.config import settings

app = FastAPI(root_path=f"/{settings.ENV}")

# ロガーの設定をロード
logger = setup_logger()

# エラーハンドラーの設定
app.add_exception_handler(Exception, create_exception_handler(logger))

# APIRouterをインクルード
app.include_router(api_router, prefix="/api")

@app.get("/env")
def health_check():
    return {"message": f"Hello from {settings.ENV} app"}

# Mangumアダプターを利用してAWS Lambda対応
handler = Mangum(app)
