# 🤝 Contributing to Chef.ai

Thank you for considering contributing! This guide will walk you through the steps to contribute effectively to this project.

---

## 🛠️ Setup Instructions

### 1. Fork the Repository

* Click on the Fork button at the top-right corner of the repository page.
* This creates a personal copy under your GitHub account.

### 2. Clone Your Fork Locally

```bash
git clone https://github.com/your-username/chefai-backend.git
cd chefai-backend
```

### 3. Add Upstream Remote

```bash
git remote add upstream https://github.com/Udhay-Adithya/chefai-backend.git
```

### 4. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```


### 5. Add Dataset
Place your original [RecipeNLG](https://www.kaggle.com/datasets/saldenisov/recipenlg/data) dataset file (recipes.csv) into the data/ folder
RecipeNLG Dataset : https://www.kaggle.com/datasets/saldenisov/recipenlg/data

---

## 🧪 Testing Your Changes

You can run individual scripts:

```bash
python src/preprocess.py
python src/embed.py
python src/search.py --query "tomato, garlic"
```

Or launch the FastAPI server:

```bash
uvicorn src.app:app --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ✅ Making a Pull Request

1. Create a new branch

```bash
git checkout -b feature/your-feature-name
```

2. Make your changes, test thoroughly, then commit:

```bash
git add .
git commit -m "Add: your change summary"
```

3. Push and create PR:

```bash
git push origin feature/your-feature-name
```

4. Go to your GitHub fork and open a Pull Request against the main branch of the original repo.

---

## 📌 Contribution Tips

* Follow PEP8 and comment your code.
* Keep commits atomic.
* Test your changes before PR.
* Use descriptive branch and commit names.

Thanks for contributing! ❤️
