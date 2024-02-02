FROM python:3.12.1-bookworm

RUN apt update
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -U sentence_transformers fastapi pymilvus minio
RUN pip install --no-cache-dir -U "uvicorn[standard]"

COPY src/base /sentence_transformers_base
RUN python3 /sentence_transformers_base/init_sentence_transformers.py


