FROM editii/sentence_transformers_base:0.1-SNAPSHOT

ENV SENTENCE_TRANSFORMERS_SERVER_CACHE_DIR=/cache

COPY src /sentence_transformers_server

# RUN useradd -m dockerapp -u 10001
USER 10001

EXPOSE 11111

WORKDIR /sentence_transformers_server
# ENTRYPOINT [ "python3", "/sentence_transformers_server/simile/restapi.py" ]
ENTRYPOINT [ "uvicorn", "src.simile.restapi:app",  "--port", "11111" ]
