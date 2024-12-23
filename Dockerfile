# ベースイメージを選択
FROM python:3.9-slim

# 必要なライブラリをインストール
RUN apt-get update && \
    apt-get install -y mecab libmecab-dev mecab-ipadic-utf8 && \
    rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# ソースコードをコピー
COPY . /app

# 依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションを起動
CMD ["python", "app.py"]
