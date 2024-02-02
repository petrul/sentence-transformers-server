FROM editii/sentence_transformers_base:0.1-SNAPSHOT

ENV SENTENCE_TRANSFORMERS_SERVER_CACHE_DIR=/cache

COPY src /sentence_transformers_server

# user dockerapp/10001 was already created in the base image
USER 10001

EXPOSE 11111

WORKDIR /sentence_transformers_server
ENV PATH="/home/dockerapp/.local/bin:${PATH}"

ENTRYPOINT [ "uvicorn", "simile.restapi:app",  "--port", "11111" ]