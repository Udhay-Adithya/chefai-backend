# 🍳 Chef.ai

A local-first AI-powered recipe recommendation system that suggests recipes based on a list of ingredients. This app uses sentence-transformer embeddings and FAISS to enable semantic recipe search. It also exposes a simple FastAPI endpoint for querying recipe suggestions.

---

## 🧠 Features

* Offline semantic search using FAISS
* Embedding generation with sentence-transformers
* Fully local, no internet or API keys required
* Optional FastAPI backend for REST queries
* Clean modular codebase

---

## 🗂 Project Structure

```
chefai-backend/
├── data/                        # Raw and processed recipe data
├── embeddings/                 # FAISS index and metadata
├── src/                        # Core scripts (preprocessing, embedding, search, API)
├── tests/                      # Basic test scripts
├── utils/                      # Helper utilities
├── requirements.txt            # Dependencies
├── README.md                   # Project overview (this file)
└── CONTRIBUTING.md             # Contribution guidelines
```

---


## 🔧 Tech Stack

* Python 3.9+
* pandas
* sentence-transformers
* faiss-cpu
* FastAPI + Uvicorn

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♀️ Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📬 Contact

For questions or suggestions, open an issue or contact @your-username on GitHub.
