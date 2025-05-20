import pandas as pd
import json
import os
from tqdm import tqdm
import math

INPUT_CSV = "data/recipes.csv"
OUTPUT_DIR = "data/processed_recipes"

def clean_text_list(lst):
    if isinstance(lst, str):
        try:
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
    total_rows = len(df)
    split_size = math.ceil(total_rows / 4)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Cleaning and splitting into 4 files...")
    for i in range(4):
        start = i * split_size
        end = min(start + split_size, total_rows)
        chunk = df.iloc[start:end]

        processed = []
        for idx, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"Processing chunk {i+1}"):
            try:
                text = combine_fields(row)
                doc = {
                    "id": int(start + idx),
                    "title": row["title"],
                    "ingredients": clean_text_list(row["ingredients"]),
                    "directions": clean_text_list(row["directions"]),
                    "source": row.get("source", "Unknown"),
                    "text": text
                }
                processed.append(doc)
            except Exception as e:
                print(f"Error processing row {row.get('id', 'unknown')}: {e}")

        output_path = os.path.join(OUTPUT_DIR, f"recipes_part_{i+1}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(processed)} recipes to {output_path}")

    print("Preprocessing and splitting complete.")

preprocess()
