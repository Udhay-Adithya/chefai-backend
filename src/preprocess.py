import pandas as pd
import json
import os
from tqdm import tqdm

INPUT_CSV = "data/recipes.csv"
OUTPUT_JSON = "data/processed_recipes.json"

def clean_text_list(lst):
    if isinstance(lst, str):
        try:
            # Convert stringified list to actual list
            lst = eval(lst)
        except:
            return []
    if not isinstance(lst, list):
        return []
    return [str(item).strip().lower() for item in lst if isinstance(item, str)]

def combine_fields(row):
    title = str(row["title"]).strip().lower()
    ingredients = clean_text_list(row["ingredients"])
    directions = clean_text_list(row["directions"])

    ingredients_text = "ingredients: " + ", ".join(ingredients)
    directions_text = "directions: " + " ".join(directions)

    return f"{title}. {ingredients_text}. {directions_text}."

def preprocess():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_CSV)

    print("Cleaning and combining fields...")
    processed = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = combine_fields(row)
            doc = {
                "id": int(_),
                "title": row["title"],
                "ingredients": clean_text_list(row["ingredients"]),
                "directions": clean_text_list(row["directions"]),
                "source": row.get("source", "Unknown"),
                "text": text
            }
            processed.append(doc)
        except Exception as e:
            print(f"Error processing row {row.get('id', 'unknown')}: {e}")

    print(f"Saving {len(processed)} processed recipes...")
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print("Preprocessing complete.")

preprocess()