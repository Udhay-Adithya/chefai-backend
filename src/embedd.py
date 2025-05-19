import ijson
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "Linq-AI-Research/Linq-Embed-Mistral"
INPUT_JSON = "data/processed_recipes.json"
OUTPUT_EMBEDDINGS = "data/recipe_embeddings.npy"
BATCH_SIZE = 4

def convert_text_to_embeddings():
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME,device="cuda")
    task = "Given a recipe description, generate a useful embedding for semantic search"
    prompt = f"Instruct: {task}\nQuery: "

    print("Streaming and embedding recipes...")
    texts = []
    embeddings = []

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        # Stream JSON array items one at a time
        objects = ijson.items(f, "item")
        for obj in tqdm(objects, desc="Processing"):
            texts.append(prompt + obj["text"])
            if len(texts) == BATCH_SIZE:
                emb = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False,normalize_embeddings=True)
                embeddings.extend(emb)
                texts = []

        # Handle remaining texts
        if texts:
            emb = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False,normalize_embeddings=True)
            embeddings.extend(emb)

    embeddings = np.array(embeddings)
    np.save(OUTPUT_EMBEDDINGS, embeddings)
    print(f"Saved {len(embeddings)} embeddings to {OUTPUT_EMBEDDINGS}")

convert_text_to_embeddings()
