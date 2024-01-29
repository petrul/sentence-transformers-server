Alias embeddings-server

This is basically a SBERT sentence_transformers 'client' which uploads data into milvus.

## Prereqs
```
$ pip install sentence_transformers
$ pip install fastapi
$ pip install "uvicorn[standard]"
$ pip install pymilvus

Or, run bin/install-deps.sh

```


## Test
Run 
``` $ rake test
```
