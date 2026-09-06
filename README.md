# ML Regression, NLP Sentiment Analysis & LLM Q&A (CA2)

## Objective
Three-part machine learning assessment covering supervised regression with hyperparameter tuning, NLP sentiment analysis, and a local open-source LLM Q&A pipeline. Developed for the Machine Learning module (CA2), L7 Diploma in Data Analytics, CCT College Dublin.

---

## Task 1 — Neural Network Regression on the Abalone Dataset

**Objective:** predict the number of rings (age) of abalone using a Neural Network and compare it against regularised linear models.

**Dataset:** 4,180 records (see `abalone.csv`) — physical measurements plus `Sex` (categorical) and `Rings` (target).

**Data Preparation:**
- `Sex` converted to numeric via One-Hot Encoding.
- Split 75% train (3,135) / 25% test (1,045).
- Features standardised (`StandardScaler`) — necessary given very different units/scales across variables.

**Models & Hyperparameter Tuning:** Neural Network (MLP Regressor), Linear, Ridge and Lasso Regression. `GridSearchCV` (5-fold) used to tune regularisation strength — best Lasso alpha = 0.001 (MSE 5.2125), best Ridge alpha = 1.0 (MSE 5.2151).

**Results (test set):**

| Model | MSE | R² |
|---|---|---|
| Linear | 4.8336 | 0.5301 |
| Ridge | 4.8321 | 0.5303 |
| Lasso | 4.8330 | 0.5302 |
| **Neural Network (MLP Regressor)** | **4.4348** | **0.5685** |

**Limitations:** Ridge and Lasso couldn't capture the non-linearity in the data — regularisation alone wasn't enough. The `Rings` distribution is concentrated between 7–15, limiting how well any model generalises to less common (older) abalones.

---

## Task 2 — Sentiment Analysis on Yelp Reviews (NLP)

**Objective:** classify review sentiment (positive/neutral/negative) from free-text Yelp reviews.

**Dataset:** `yelp.csv`, 10,000 reviews. `stars` converted into 3 sentiment classes; `cool`/`useful`/`funny` used as engagement features. Class distribution: 5,490 positive, 1,341 negative, 1,169 neutral (imbalanced — handled with stratified 80/20 train/test split).

**Preprocessing pipeline (reusable, built in `preprocess_text`):** punctuation/URL/special-character removal, lowercasing, stop-word removal, tokenisation, and lemmatisation (WordNetLemmatizer).

**Approach & Results:**
- **VADER** (lexicon-based, no training needed): 0.7305 accuracy — struggled specifically with the neutral class (low recall).
- **Bag of Words + Logistic Regression:** **0.7825 accuracy — best overall.**
- **TF-IDF + Logistic Regression:** 0.7725 accuracy.

| Model | Accuracy |
|---|---|
| BoW + Logistic Regression | 0.7825 |
| TF-IDF + Logistic Regression | 0.7725 |
| VADER | 0.7305 |

**Insight:** the simpler Bag-of-Words model outperformed TF-IDF here — suggesting that for these reviews, whether a word appears at all matters more than how rare it is.

---

## Task 3 — LLM Q&A Pipeline on Banking Text (Generative AI)

**Objective:** apply a local, open-source LLM to generate and answer business/banking questions from a text corpus, and reflect on the ethical risks of doing so.

**Source text:** `pg32027.txt` (242,340 characters), cleaned to 237,325 characters and split into 238 overlapping chunks (1,200 characters + 200-character overlap, to preserve context across chunk boundaries), then vectorised with TF-IDF.

**Pipeline (`llmfunctions.py`):** 5 chunks sampled to build a topic-guided corpus summary; a prompt then asks the model to generate 5 business/banking questions (with 5 fallback questions as a safety net). Several `google/flan-t5` variants were tested; `google/flan-t5-base` was selected for comparable quality at lower compute cost.

**Results:**
- Only **1 of 5** questions were genuinely model-generated — the other 4 fell back to the pre-written fallback questions.
- The model correctly returned *"Not enough evidence in the provided text"* when appropriate.
- **Hallucination observed twice**, and a **context-mismatch error once** — the built-in fallback/mitigation logic triggered as designed in those cases.

**Limitations:** a small, general-purpose model (`flan-t5-base`) run locally has real reliability limits for grounded Q&A — most of the "5 generated questions" target wasn't met, and hallucination is a known risk that any real banking application of this pipeline would need stronger guardrails and human review for.

---

## Conclusion
Across the three tasks: non-linear models (Neural Network) meaningfully outperform regularised linear models when the underlying relationship isn't linear (Task 1); simpler NLP techniques can beat more "sophisticated" ones depending on the text data (Task 2); and small open-source LLMs run locally are useful for prototyping RAG-style Q&A but need explicit fallback handling and human oversight before any real banking use, given the observed hallucination rate (Task 3).

## Files in this repository

| File | Description |
|---|---|
| [`Code.ipynb`](CA_2_ML_sba25076.ipynb) | Full analysis notebook — all 3 tasks (regression, NLP, LLM pipeline) |
| [`PDF Report.pdf`](<./PDF Report.pdf>) | Full written report covering all 3 tasks |
| [`abalone.csv`](./abalone.csv) | Dataset for Task 1 (regression) |
| [`yelp.csv`](./yelp.csv) | Dataset for Task 2 (sentiment analysis) |
| [`pg32027.txt`](./pg32027.txt) | Source text corpus for Task 3 (LLM Q&A) |
| [`llmfunctions.py`](./llmfunctions.py) | Helper functions for the Task 3 LLM pipeline (chunking, summary, Q&A generation) |
