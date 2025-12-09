# Myanmar Agri-Assist: AI-Powered Agricultural Knowledge Assistant

Myanmar Agri-Assist is an end-to-end Burmese-language agricultural knowledge assistant. The system ingests a dataset with the following columns:

- **Instruction** - farmer query or guidance in Burmese.
- **Output** - authoritative answer or agronomic recommendation.
- **Category** - manually curated topic label (for example soil, irrigation, crop protection).

Every text preprocessing step relies on the [pyidaungsu](https://pypi.org/project/pyidaungsu/) tokenizer to properly segment Burmese script before training, keyword extraction, and embedding generation.

## Repository Structure

```
project/
|-- data/Agriculture.csv
|-- eda/eda_analysis.py
|-- model/train_classifier.py
|-- model/embeddings_rag.py
|-- api/app.py
|-- api/rag_engine.py
|-- web/interface.py
|-- requirements.txt
`-- README.md
```

### High-Level Architecture

```
[Agriculture CSV] --> [EDA + Keywords] --> [Topic Classifier]
        |                                 /
        |                                /
        |                       [Burmese RAG Index]
        |                               |
[FastAPI Backend] <--> [Streamlit Interface]
```

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Exploratory Data Analysis**

   ```bash
   python eda/eda_analysis.py
   ```

   This generates:
   - `eda/category_pie.png`
   - `eda/category_bar.png`
   - `eda/category_keywords.json`

3. **Train the Topic Classifier**

   ```bash
   python model/train_classifier.py
   ```

   The script tokenizes Burmese text with pyidaungsu, builds a TF-IDF representation, trains a Logistic Regression classifier, and saves `model/topic_model.pkl` plus `model/classification_report.txt`.

4. **Build the RAG Index (optional)**

   The first time you run anything that imports `model/embeddings_rag.py`, the FAISS index and metadata store are generated automatically:
   - `model/faiss_index.bin`
   - `model/faiss_metadata.json`

## Running the Applications

- **FastAPI Backend**

  ```bash
  uvicorn api.app:app --host 0.0.0.0 --port 8000
  ```

  Endpoints:
  - `POST /classify` -> `{ "category": "...", "confidence": 0.95 }`
  - `POST /ask` -> `{ "answer": "...", "contexts": [...] }`

- **Streamlit Interface**

  ```bash
  streamlit run web/interface.py
  ```

  Tabs:
  1. **Dataset Overview** - displays sample rows, pie and bar charts, and TF-IDF keyword JSON.
  2. **Topic Classification** - accepts Burmese text and shows the predicted category with probability.
  3. **RAG Assistant** - answers Burmese agricultural questions by retrieving the top knowledge snippets.

## Burmese Text Processing Notes

- All scripts call `pyidaungsu.tokenize(text, form="word")` to produce Burmese word tokens.
- Tokens are lowercased and re-joined with whitespace to preserve compatibility with scikit-learn TF-IDF vectorizers and SentenceTransformer embeddings.
- Downstream features (EDA, classifier, embeddings) share the exact preprocessing routine for consistency.

## Troubleshooting

- Missing plots or keyword files? Re-run `eda/eda_analysis.py`.
- Backend complains about `topic_model.pkl`? Run `python model/train_classifier.py`.
- `faiss` errors? Ensure the correct version (`faiss-cpu`) is installed and rebuild the index by deleting `model/faiss_index.bin`.

Happy farming with Myanmar Agri-Assist!
