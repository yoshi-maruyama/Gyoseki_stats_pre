import os
from dotenv import load_dotenv

# ベースディレクトリのパスを取得
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# デフォルトの.envファイルを読み込む
load_dotenv(os.path.join(BASE_DIR, ".env"))

# 環境に応じた.envファイルを読み込む
env = os.getenv("ENV", "dev")
env_file = os.path.join(BASE_DIR, f".env.{env}")
if os.path.exists(env_file):
    load_dotenv(env_file, override=True)
else:
    print(f"Warning: .env.{env} file not found. Using default settings.")

# 設定値を定義
class Settings:
    ENV: str = os.getenv("ENV", "dev")
    BASIC_USERNAME: str = os.getenv("STATS_BASIC_USERNAME")
    BASIC_PASSWORD: str = os.getenv("STATS_BASIC_PASSWORD")

settings = Settings()
