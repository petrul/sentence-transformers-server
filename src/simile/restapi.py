from fastapi import FastAPI, HTTPException
from .encoder import *

app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}



all_MiniLM_L6_v2 =EncoderFactory.all_MiniLM_L6_v2()
models = {
    'all_MiniLM_L6_v2': all_MiniLM_L6_v2
}

def p(*args): print(*args)

@app.post("/api/models/{model_id}/encode")
def post_encode_(model_id: str, body: list[str]):
    if not model_id in models.keys():
        raise HTTPException(status_code=404, detail="Model not found")
    encoder = models[model_id]
    resp = encoder.encode(body)
    return resp.tolist()
