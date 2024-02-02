FROM editii/sentence_transformers_base:0.1-SNAPSHOT

ENV SENTENCE_TRANSFORMERS_SERVER_CACHE_DIR=/cache

# RUN useradd -m dockerapp -u 10001
USER 10001

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -U sentence_transformers fastapi pymilvus minio
RUN pip install --no-cache-dir -U "uvicorn[standard]"

COPY src /sentence_transformers_server

ENTRYPOINT [ "python3", "/sentence_transformers_server/simile/restapi.py" ]