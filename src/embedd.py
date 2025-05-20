import os
import ijson
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
INPUT_DIR = "data/processed_recipes"  # Directory with recipes_part_*.json files
OUTPUT_DIR = "data/embeddings_chunks"
BATCH_SIZE = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_text_to_embeddings():
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    task = "Given a recipe description, generate a useful embedding for semantic search"
    prompt = f"Instruct: {task}\nQuery: "

    print("Checking existing chunks...")
    existing_chunks = sorted([
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("chunk_") and f.endswith(".npy")
    ])
    start_chunk = max(existing_chunks) + 1 if existing_chunks else 0
    print(f"Last completed chunk: {start_chunk - 1 if start_chunk > 0 else 'None'}")

    chunk_idx = 0
    item_idx = 0
    texts = []

    json_files = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".json") and f.startswith("recipes_part_")
    ])

    for json_file in json_files:
        print(f"Processing file: {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            objects = ijson.items(f, "item")

            for obj in tqdm(objects, desc=f"Processing {os.path.basename(json_file)}"):
                if chunk_idx < start_chunk:
                    item_idx += 1
                    if item_idx % BATCH_SIZE == 0:
                        chunk_idx += 1
                    continue  # skip previously processed chunks

                texts.append(prompt + obj["text"])

                if len(texts) == BATCH_SIZE:
                    emb = model.encode_multi_process(texts, pool_size=os.cpu_count(), normalize_embeddings=True, show_progress_bar=False)
                    np.save(os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx}.npy"), emb)
                    texts = []
                    chunk_idx += 1

    if texts:  # Final incomplete batch
        emb = model.encode_multi_process(texts, pool_size=os.cpu_count(), normalize_embeddings=True, show_progress_bar=False)
        np.save(os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx}.npy"), emb)

    print(f"Saved up to chunk {chunk_idx} in {OUTPUT_DIR}")

convert_text_to_embeddings()
