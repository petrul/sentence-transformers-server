FROM python:3.12.1-bookworm

RUN apt update
RUN pip install sentence_transformers
RUN pip install fastapi
RUN pip install "uvicorn[standard]"
RUN pip install pymilvus
RUN pip install minio

COPY src/base /sentence_transformers_base
RUN python3 /sentence_transformers_base/init_sentence_transformers.py


