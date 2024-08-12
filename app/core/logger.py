import logging

def setup_logger():
    # ロガーの設定
    logger = logging.getLogger("gyoseki_stats")
    logger.setLevel(logging.INFO)  # ログレベルをINFOに設定

    # Lambda環境では、標準のログストリームが使われるため、特別な設定は不要
    # ストリームハンドラーを追加
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # フォーマッターを設定
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # ハンドラーをロガーに追加
    logger.addHandler(handler)

    return logger
