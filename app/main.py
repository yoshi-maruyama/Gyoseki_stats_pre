## app/main.py

from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/item")
async def read_item():
    return {"item": "hello world"}

# Mangumアダプターを利用してAWS Lambda対応
handler = Mangum(app)
