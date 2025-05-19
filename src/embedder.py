from sentence_transformers import SentenceTransformer
import json
import numpy as np

MODEL_NAME = "Linq-AI-Research/Linq-Embed-Mistral"
INPUT_JSON = "data/processed_recipes.json"
OUTPUT_EMBEDDINGS = "data/recipe_embeddings.npy"

def convert_text_to_embeddings():
    # Load model
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    # Define the task and prompt
    task = "Given a recipe description, generate a useful embedding for semantic search"
    prompt = f"Instruct: {task}\nQuery: "

    # Load preprocessed recipes
    print("Loading recipes...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        recipes = json.load(f)

    # Extract texts
    texts = [prompt + recipe["text"] for recipe in recipes]

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=4)

    # Save embeddings as .npy
    np.save(OUTPUT_EMBEDDINGS, embeddings)
    print(f"Saved {len(embeddings)} embeddings to {OUTPUT_EMBEDDINGS}")

convert_text_to_embeddings()