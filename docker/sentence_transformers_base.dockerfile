FROM python:3.12.1-bookworm

ENV SENTENCE_TRANSFORMERS_SERVER_CACHE_DIR=/cache

RUN apt-get --allow-releaseinfo-change update
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y apt-utils build-essential cmake pkg-config
RUN apt install -y python3-pkgconfig 

RUN useradd -m dockerapp -u 10001
USER 10001

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -U sentence_transformers fastapi pymilvus minio
RUN pip install --no-cache-dir -U "uvicorn[standard]"

COPY src/base /sentence_transformers_base
RUN python3 /sentence_transformers_base/init_sentence_transformers.py


