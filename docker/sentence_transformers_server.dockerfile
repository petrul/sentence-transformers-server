FROM editii/sentence_transformers_base:0.1-SNAPSHOT

ENV SENTENCE_TRANSFORMERS_SERVER_CACHE_DIR=/cache

COPY src /sentence_transformers_server

# user dockerapp/10001 was already created in the base image
USER 10001

EXPOSE 8000

WORKDIR /sentence_transformers_server/simile
ENV PATH="/home/dockerapp/.local/bin:${PATH}"

ENTRYPOINT [ "uvicorn", "restapi:app", "--host", "0.0.0.0" ]