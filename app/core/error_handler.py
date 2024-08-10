import traceback
from fastapi import Request
from fastapi.responses import JSONResponse

def create_exception_handler(logger):
    async def custom_exception_handler(request: Request, exc: Exception):
        # エラーメッセージとトレースバックをログに記録
        error_message = f"Exception occurred: {str(exc)}"
        stack_trace = traceback.format_exc()
        logger.error(f"{error_message}\n{stack_trace}")

        # エラーレスポンスを返す
        return JSONResponse(
            status_code=500,
            content={"message": f"{error_message}"},
        )

    return custom_exception_handler
