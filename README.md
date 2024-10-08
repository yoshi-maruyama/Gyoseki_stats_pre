# Gyoseki Stats

Gyosekicom における統計情報計算処理を行うための FastApi サーバー

## 環境構築

```
docker compose up --build
```

localhost:8000/docs
を開いて swagger の画面が表示されることを確認してください。

画面を開く際に basic 認証を求められると思いますが,
下記で突破できるようになります。

```
username→user
password→password
```

## デバッグ

pdb を使ってブレークポイントを設定することができます。

```python
router = APIRouter()

class Stats(BaseModel):
    id: str

@router.get("/", response_model=Stats)
def read_stats(stats_service: StatsService = Depends()) -> Any:
    """
    Retrieve string
    """
    hoge = stats_service.create_matrix()
    import pdb; pdb.set_trace() ## ここ
    return Stats(id=hoge)
```

この処理が実行されると処理が止まります。
docker attach gyosekicom_stats-app-1
でコンテナにアタッチして検証できます。

```
> /app/app/api/routes/stats.py(19)read_stats()
-> return Stats(id=hoge)
(Pdb)
```

となります。
ここでいろいろ動作させることができます。

## デプロイ

デプロイ先は AWS lambda に直接あげます。

gyoseki-stats-ecr リポジトリへ docker イメージをプッシュ
gyoseki-stats lambda 関数で先の ECR の latest をアーキテクチャ arm64 でデプロイする
関数 URL を発行する
その URL で/ を叩く

```
staging

https://rft7t2ado2onqvjgovsoyayisq0nixso.lambda-url.ap-northeast-1.on.aws/
https://rft7t2ado2onqvjgovsoyayisq0nixso.lambda-url.ap-northeast-1.on.aws/docs
```

環境変数には下記項目を設定してください

```
ENV
STATS_BASIC_USRNAME
STATS_BASIC_PASSWORD
```
