from fastapi import FastAPI, HTTPException
from encoder import *
from util import *
from application_properties import *

app = FastAPI()

appprops = ApplicationProperties()
cachedir = appprops.cacheDir()
encoderFactory = EncoderFactory(cacheRootDir=cachedir)

@app.get("/api/models/names")
def get_model_names():
    return [NAME_ALL_MINILM_L6_V2, NAME_ALL_MPNET_BASE_V2]

@app.post("/api/models/{model_id}/encode")
def post_encode_(body: list[str], model_id: str = NAME_ALL_MINILM_L6_V2):
    # if not model_id in models.keys():
    #     raise HTTPException(status_code=404, detail="Model not found")
    enc = encoderFactory[model_id]
    # encoder = models[model_id]
    resp = enc.encode(body)
    return [ it.tolist() for it in resp]
