# Disinformation Detection on Social Media: Russia–Ukraine War

A text classification project that detects disinformation in English-language social media posts and news headlines about the Russia–Ukraine war. It covers the full workflow, from data cleaning to model tuning, error analysis and an interactive Streamlit dashboard.

> Team capstone project, **SoftITo Data Analytics Program** (2026)
> Notebooks, comments and the dashboard interface are in Turkish. This README gives an English overview.

---

## Results at a glance

| | |
|---|---|
| **Dataset** | 7,957 labeled texts → 7,586 after cleaning |
| **Best model** | Optimized XGBoost: **96.9% accuracy** |
| **Baseline** | Majority class (always "disinformation"): 72.5% accuracy |
| **Minority-class recall** | 0.98 (non-disinformation texts) |
| **Models compared** | 6 classifiers, top 3 tuned with GridSearchCV |

---

## Dataset

| Feature | Value |
|---|---|
| Records | 7,957 (no missing values, no duplicates) |
| Columns | `post_id`, `text`, `is_disinformation`, `news_type`, `confidence` |
| Class balance (raw) | yes: 5,552 (69.8%) · no: 2,405 (30.2%) |
| Dominant category | Politics (75.3%) |

Because the classes are imbalanced, the project tracks **recall on the minority ("no") class** alongside accuracy, since accuracy alone would be misleading.

---

## Workflow

The analysis is split into seven sequential notebooks:

| # | Notebook | What it does |
|---|---|---|
| 1 | Data cleaning & EDA | Encoding repair, language/script filtering, off-topic removal, distributions |
| 2 | Feature engineering | Stratified 80/20 split, TF-IDF vectorization, vocabulary checks |
| 3 | Model training & comparison | Baseline + six classifiers evaluated on accuracy, precision, recall, F1 and confusion matrices |
| 4 | Hyperparameter optimization | GridSearchCV (5-fold, macro F1) and a train vs. test overfitting check |
| 5 | Threshold optimization & error analysis | Decision-threshold tuning on cross-validated predictions; false positive / false negative review |
| 6 | Research questions | Topic, text length, vocabulary, n-gram and sentiment analysis |
| 7 | Google Trends | Contextual validation of model findings against search-interest trends |

### 1. Data cleaning

A four-step cleaning pipeline reduced the data from 7,957 to **7,586 records**:

- **Encoding repair:** A custom `fix_mojibake()` function repaired texts where UTF-8 had been decoded as cp1252, which mainly affected Cyrillic, Chinese and emoji content.
- **Script detection:** `langdetect` misclassified short English headlines (for example, as Norwegian), so a custom `non_latin_ratio()` function was written instead. A 16% threshold cleanly separated genuinely foreign-language texts (ratio 0.63–0.79) from English texts containing a few foreign words (around 0.07–0.15).
- **Off-topic filtering:** 515 candidate records about other conflicts and topics (Gaza, Iran, Taiwan, COVID-19 and others) were screened. 356 were removed, and 2 were kept after manual review.
- **Short-text removal:** 12 truncated records under 25 characters were dropped.

### 2. Features

- `TfidfVectorizer(stop_words="english", min_df=3, max_df=0.9, ngram_range=(1, 2))`
- 11,143 features, reduced to **11,037** after removing number-only tokens. Meaningful number–word pairs such as "000 soldiers" were kept.

### 3. Models

| Model | Accuracy | "No" recall | "No" precision |
|---|---|---|---|
| Baseline (majority class) | 0.725 | 0.00 | 0.00 |
| Multinomial Naive Bayes | 0.854 | 0.49 | 0.97 |
| Complement Naive Bayes | 0.912 | 0.77 | 0.89 |
| Logistic Regression | 0.922 | 0.75 | 0.96 |
| **Optimized Linear SVM** | 0.964 | 0.94 | 0.93 |
| **Optimized Random Forest** | 0.969 | 0.96 | 0.93 |
| **Optimized XGBoost** | **0.969** | **0.98** | 0.91 |

Adding `class_weight="balanced"` clearly improved minority-class recall for SVM and Random Forest. Because missing disinformation is usually the costlier error, **XGBoost** is the recommended final model.

### 4. Error analysis

| Model | TN | FP | FN | TP | Total errors |
|---|---|---|---|---|---|
| Linear SVM | 395 | 23 | 32 | 1,068 | 55 |
| Random Forest | 402 | 16 | 33 | 1,067 | 49 |
| XGBoost | 410 | 8 | 39 | 1,061 | 47 |

- Misclassified texts are about **half the length** of correctly classified ones (around 140–150 vs. 280–310 characters). The pattern holds across all models, which points to truncated source texts.
- False positives are often triggered by words such as *claim*, *denied* and *allegedly*, which legitimate news also uses when reporting allegations.

---

## Key findings

1. **Disinformation texts are about 4× longer** than genuine news (356 vs. 82 characters on average, Mann–Whitney U, p ≈ 0). A model that uses text length alone reaches 92.5% accuracy.
2. **Three independent model families agree** on the most predictive words (*propaganda, fake, claims, allegedly, west/western*), so the signal is not a random artifact.
3. **Known propaganda framing appears in the n-grams,** for example *"special military operation"* and *"Kyiv regime"*.
4. **Disinformation is more negative in tone:** the median VADER score is −0.477, compared with −0.128 for genuine news (p ≈ 7 × 10⁻⁵⁴).
5. **Topic matters less than expected:** Politics, Technology and Other make up 98% of the data and have similar disinformation rates (68–74%). Education is the exception at 14%.

## Limitations

- Part of the model's success may come from **surface patterns** such as text length, fact-check meta-language and source-specific templates, rather than a real understanding of the content.
- Several categories (Health, Science, Business) have fewer than 10 test records, so results for them are not reliable.
- The `confidence` label barely varies (93% of rows are 0.7), so its relationship with model performance could not be tested.

**Next steps:** length-controlled evaluation, leave-source-out cross-validation, and more labeled data for small categories.

---

## Interactive dashboard

`app.py` is a Streamlit dashboard with sections for overview, data exploration, cleaning, modeling and optimization, feature analysis, conclusions, and a **live prediction** page where you can paste text and classify it. The interface is in Turkish.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository structure

```
├── 1-veri temizleme ve keşifsel analiz/   # Data cleaning & EDA
├── 2-özellik mühendisliği/                 # Feature engineering
├── 3-model eğitimi ve karşılaştırma/       # Model training & comparison
├── 4-hiperparametre optimizasyonu/         # Hyperparameter optimization
├── 5-esik optimizasyonu ve hata analizi/   # Threshold optimization & error analysis
├── 6-araştırma soruları/                   # Research questions
├── 7-google trends/                        # Google Trends validation
├── veri setimiz/                           # Dataset (veri_seti_duzenlendi.xlsx)
├── app.py                                  # Streamlit dashboard
└── requirements.txt
```

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · NLTK (VADER) · Matplotlib · Plotly · WordCloud · Streamlit · Jupyter

## Team

- Derya Kalay
- Feza Nur Hazar
- İrem Nur
- Rabia Ekinci
- Sümeyya Taş
- Şevval Birgül
