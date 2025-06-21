
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.model import load_model
from src.search import search_recipes


app = FastAPI()
model = load_model()

class SearchRequest(BaseModel):
    ingredients: str
    top_k: int = 5
@app.get("/")
def root():
    return {"message": "👋 Welcome to ChefAI API! Use /docs to try it out."}

@app.post("/search")
def search(request: SearchRequest):
    try:
        prompt = "Instruct: Given a recipe description, generate a useful embedding for semantic search\nQuery: "
        full_query = prompt + request.ingredients
        embedding = model.encode(full_query, normalize_embeddings=True)
        results = search_recipes(embedding.tolist(), top_k=request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
