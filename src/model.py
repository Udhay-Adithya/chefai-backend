from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

def load_model():
    return SentenceTransformer(MODEL_NAME)
