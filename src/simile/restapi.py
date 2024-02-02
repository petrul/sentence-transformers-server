from fastapi import FastAPI, HTTPException
from .encoder import *
from .util import *

app = FastAPI()


# NAME_ALL_MINILM_L6_V2= 'all_MiniLM_L6_v2'
# NAME_ALL_MPNET_BASE_V2='all-mpnet-base-v2'



@app.post("/api/models/{model_id}/encode")
def post_encode_(body: list[str], model_id: str = NAME_ALL_MINILM_L6_V2):
    # if not model_id in models.keys():
    #     raise HTTPException(status_code=404, detail="Model not found")
    encoder = models[model_id]
    resp = encoder.encode(body)
    return [ it.tolist() for it in resp]
