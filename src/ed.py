import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import orjson

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
INPUT_JSON = "data/processed_recipes/recipes_part_1.json"
OUTPUT_PATH = "data/embeddings_chunks/recipes_part_1_embeddings.npz"
BATCH_SIZE = 128  # Still useful if you chunk later

def main():
    print("Loading model for multi-process encoding...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loading data from {INPUT_JSON}...")
    with open(INPUT_JSON, "rb") as f:
        data = orjson.loads(f.read())

    print("Preparing texts...")
    task = "Given a recipe description, generate a useful embedding for semantic search"
    prompt = f"Instruct: {task}\nQuery: "
    texts = [prompt + doc["text"] for doc in tqdm(data)]

    print("Encoding embeddings using multi-process...")
    embeddings = model.encode_multi_process(texts,pool = model.start_multi_process_pool(), normalize_embeddings=True)

    print(f"Saving to {OUTPUT_PATH}...")
    np.savez_compressed(OUTPUT_PATH, embeddings=embeddings)
    print("Done!")

if __name__ == "__main__":
    main()
