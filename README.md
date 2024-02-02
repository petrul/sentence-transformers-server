# Intro 
Alias embeddings-server

This is basically a sentence_transformers 'client' which makes the sentence_transformers python
api available over a RESTI API.

- uploads data into milvus.

## Prereqs
```
$ pip install sentence_transformers
$ pip install fastapi
$ pip install "uvicorn[standard]"
$ pip install pymilvus
$ pip install minio

```

Alternatively, run bin/install-deps.sh.

## Test
Run 
``` $ rake test
```
