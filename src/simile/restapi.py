from fastapi import FastAPI, HTTPException
from encoder import *
from util import *
from application_properties import *
import huggingface_hub

app = FastAPI()

appprops = ApplicationProperties()
cachedir = appprops.cacheDir()
cacheFactory = VectorCacheFactory(cacheRootDir=cachedir)
encoderFactory = EncoderFactory(cacheFactory=cacheFactory)

@app.get("/api/models/names")
def get_model_names():
    return [NAME_ALL_MINILM_L6_V2, NAME_ALL_MPNET_BASE_V2]


@app.get("/api/models/{model_id}/info")
def get_models_id_info(model_id: str):
    try:
        enc = encoderFactory[model_id]
        st = enc.st
        return {
            'name': model_id,
            'max_seq_length': st.get_max_seq_length(),
        }
    
    except huggingface_hub.utils._errors.RepositoryNotFoundError as  err:
        raise HTTPException(status_code=404, detail=str(err))
    except huggingface_hub.utils._errors.HfHubHTTPError as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.post("/api/models/{model_id}/encode")
def post_models_encode_(body: list[str], model_id: str):
    try:
        enc = encoderFactory[model_id]
        resp = enc.encode(body)
        return [ it.tolist() for it in resp]
    except huggingface_hub.utils._errors.RepositoryNotFoundError as  err:
        raise HTTPException(status_code=404, detail=str(err))
    except huggingface_hub.utils._errors.HfHubHTTPError as err:
        raise HTTPException(status_code=500, detail=str(err))
