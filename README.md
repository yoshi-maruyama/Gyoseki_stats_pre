# Gyoseki Stats

Gyosekicom における統計情報計算処理を行うための FastApi サーバー

## 環境構築

```
docker compose up --build
```

## デプロイ

デプロイ先は AWS lambda に直接あげます。

gyoseki-stats-ecr リポジトリへ docker イメージをプッシュ
gyoseki-stats lambda 関数で先の ECR の latest をアーキテクチャ arm64 でデプロイする
関数 URL を発行する
その URL で/item を叩く

```
staging

https://rft7t2ado2onqvjgovsoyayisq0nixso.lambda-url.ap-northeast-1.on.aws/item
https://rft7t2ado2onqvjgovsoyayisq0nixso.lambda-url.ap-northeast-1.on.aws/docs
```
