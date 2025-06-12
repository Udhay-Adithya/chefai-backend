import os
import ijson
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

# Constants
BATCH_SIZE = 128
RECIPE_DIR = "data/processed_recipes"
EMBEDDING_DIR = "data/embeddings_chunks"
DB_CONN = {
    'dbname': 'cookai',
    'user': 'postgres',
    'password': 'root',
    'host': 'localhost',
    'port': 5432,
}

# Connect to PostgreSQL
conn = psycopg2.connect(**DB_CONN)
cur = conn.cursor()

# Ensure pgvector extension and table are ready
cur.execute("""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS recipes (
    id SERIAL PRIMARY KEY,
    title TEXT,
    ingredients TEXT,
    directions TEXT,
    embedding vector(768)
);
""")
conn.commit()

# Sort embedding chunks
embedding_files = sorted(
    [f for f in os.listdir(EMBEDDING_DIR) if f.endswith(".npy")],
    key=lambda x: int(x.split("_")[1].split(".")[0])
)

# Stream recipes and insert with corresponding embeddings
recipe_stream = []
json_files = sorted([os.path.join(RECIPE_DIR, f) for f in os.listdir(RECIPE_DIR) if f.endswith(".json")])

def get_all_recipes():
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            for item in ijson.items(f, "item"):
                yield item

recipe_iterator = get_all_recipes()
chunk_index = 0

while True:
    batch = []
    try:
        for _ in range(BATCH_SIZE):
            recipe = next(recipe_iterator)
            batch.append(recipe)
    except StopIteration:
        break  # End of data

    if not batch:
        break

    # Load corresponding embedding
    embedding_path = os.path.join(EMBEDDING_DIR, f"chunk_{chunk_index}.npy")
    embeddings = np.load(embedding_path)

    assert len(batch) == len(embeddings), f"Mismatch in batch {chunk_index}"

    # Prepare data for insertion
    records = [
    (r["title"], r["ingredients"], r["directions"], f'[{", ".join(map(str, emb))}]')
    for r, emb in zip(batch, embeddings)
]

    execute_values(
    cur,
    "INSERT INTO recipes (title, ingredients, directions, embedding) VALUES %s",
    records,
    template="(%s, %s, %s, %s::vector)"
)
    conn.commit()

    print(f"Inserted chunk {chunk_index} ✅")
    chunk_index += 1

# Final cleanup
cur.close()
conn.close()
