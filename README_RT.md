# Predicting the Tomatometer — Rotten Tomatoes ML Project

A machine learning case study that classifies movies as **Fresh** or **Rotten** based on structured metadata, using the Rotten Tomatoes dataset.

**Authors:** Ana Claudia Russo & Jeronimo Cagliero
**Program:** Ironhack, Data Analytics Bootcamp — Sep 2026

---

## Overview

Rotten Tomatoes aggregates movie ratings from critics and audiences worldwide. This project explores whether a movie's **Fresh** or **Rotten** classification can be predicted from its metadata — genre, cast, crew, runtime, and release information — before relying on aggregated review scores.

- **Goal:** Predict whether a movie is Fresh or Rotten using ML, based on metadata and audience/critic signals.
- **ML Task:** Binary classification, trained on historical rating patterns.
- **Why it matters:** Reveals which factors drive critical reception — valuable for studios, marketing teams, and streaming platforms deciding how to position content before release.

---

## Dataset

- **Source:** `rotten_tomatoes_movies.csv` — scraped from Rotten Tomatoes in October 2020, available on Kaggle: https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset?resource=download
- **Original size:** 17,712 rows × 22 columns.
- **Target:** `tomatometer_status`, converted to a binary label:
  - **Fresh** — 60% or more positive critic reviews
  - **Rotten** — less than 60% positive critic reviews

### Data preparation

1. **Filter by popularity** — kept only movies with `audience_count > 3,500` reviews, to focus on titles with statistically meaningful review volume.
2. **Categorize & cluster** — high-cardinality columns (e.g. `genre`, `content_rating`) were grouped into manageable categories with AI-assisted labeling. `directors`, `authors`, `actors`, and `production_companies` were grouped with **K-Means clustering**, based on weighted average audience rating, audience count, and movie count.
3. **Binary target** — `tomatometer_status` converted to Fresh / Rotten as described above.

**Final dataset:** 8,934 rows × 10 columns:

| Column | Description |
|---|---|
| `movie_title` | Identifier only — excluded from model features (leakage-prone) |
| `streaming_release_date` | Year of streaming release |
| `runtime` | Movie runtime, in minutes |
| `content_group` | Categorized content rating |
| `main_genre` | Categorized primary genre |
| `directors_cluster` | K-Means cluster of the movie's director(s) |
| `authors_cluster` | K-Means cluster of the movie's author(s)/writer(s) |
| `actors_cluster` | K-Means cluster of the movie's lead cast |
| `production_company_cluster` | K-Means cluster of the production company |
| `target` | Binary label derived from `tomatometer_status` (1 = Fresh, 0 = Rotten) |

---

## Feature Engineering & Selection

- **Train/test split** with stratification, applied before any transformation, to preserve class balance.
- **One-Hot Encoding** for categorical variables, with consistent dummy columns enforced across train and test sets.
- **MinMaxScaler** normalization for numerical features (e.g. `runtime`), to stabilize ensemble methods.
- **Test set reindexing** to match the training feature columns, preventing feature-mismatch errors at prediction time.
- **Feature selection:** removed non-predictive/leakage-prone columns (`movie_title`); kept `streaming_release_date`, `runtime`, `content_group`, `main_genre`, and the director/author/actor/production-company clusters, based on domain relevance.

---

## Feature Validation — Decision Tree

Before comparing ensemble models, a **Decision Tree Classifier** was used to sanity-check which features best separate Fresh from Rotten movies:

- **Root split** (`directors_cluster`) confirmed that clustering-based features are the strongest early predictors.
- `authors_cluster`, `actors_cluster`, and `content_group` followed as the next most informative splits — reinforcing the feature engineering choices.
- `main_genre` and `runtime` refined predictions within subgroups, but carried less overall weight than the clustering features.

---

## Model Building & Evaluation

Three models were trained and compared:

| Model | Train Accuracy | Test Accuracy | Notes |
|---|---|---|---|
| Random Forest | 83.75% | 73.65% | Strong training fit but a large train/test gap — clear overfitting. |
| XGBoost (original config) | ≈ 80% | ≈ 72–73% | Noticeable overfitting; used as a baseline for boosting behavior. |
| **XGBoost (manually tuned)** | **77.39%** | **74.37%** | Best balance of accuracy and generalization — selected as the final model. |

### Final model: manually tuned XGBoost

The final hyperparameters were reached through **manual experimentation** — for example, testing `max_depth` values of 3, 4, and 5, and keeping the value with the best result:

```python
xgb_tuned = XGBClassifier(
    objective="binary:logistic",   # binary classification (Fresh vs Rotten)
    eval_metric="logloss",         # evaluation metric used during training
    max_depth=3,                   # best generalization of the values tested
    learning_rate=0.1,             # moderate learning rate for stable boosting
    n_estimators=300,              # more boosting rounds for better learning
    subsample=0.8,                 # 80% of rows per tree, improves generalization
    colsample_bytree=0.8,          # 80% of features per tree, reduces variance
    reg_alpha=0.5,                 # L1 regularization, reduces model complexity
    reg_lambda=1.0,                # L2 regularization, for stability
    tree_method="hist",            # fast histogram-based algorithm
    random_state=SEED,             # reproducibility
)
```

**Test set performance:**

| Metric | Class 0 (Rotten) | Class 1 (Fresh) |
|---|---|---|
| Precision | 0.75 | 0.74 |
| Recall | 0.67 | 0.81 |
| F1-score | 0.71 | 0.77 |

- **Test Accuracy:** 74.37%
- **Train Accuracy:** 77.39%
- **Train/test gap:** ~3 points — minimal overfitting compared to the other two models.

---

## 🛠️ Tools & Libraries

**Language:** 
- Python

**Data handling & visualization:**
- pandas, numpy — data manipulation and numerical operations
- matplotlib, seaborn — exploratory data analysis and the Decision Tree visualization

**scikit-learn:**
- preprocessing — StandardScaler, MinMaxScaler
- cluster — KMeans
- model_selection — train_test_split, GridSearchCV, RandomizedSearchCV
- tree — DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
- ensemble — RandomForestClassifier, RandomForestRegressor
- neighbors — KNeighborsClassifier
- linear_model — SGDClassifier
- metrics — accuracy_score, classification_report, confusion_matrix

**Gradient Boosting:**
- xgboost — XGBClassifier, XGBRegressor

