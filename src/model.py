from fastembed import TextEmbedding

class FastEmbedModelWrapper:
    def __init__(self):
        from chefai.core.config import get_settings
        settings = get_settings()
        self.model = TextEmbedding(model_name=settings.dense_model)

    def encode(self, text: str, normalize_embeddings: bool = True):
        embeddings = list(self.model.embed([text]))
        return embeddings[0]

def load_model():
    return FastEmbedModelWrapper()
