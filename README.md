# 🚀 GitHub Project Success Predictor

A machine learning project that predicts whether a GitHub repository will reach **100+ stars** using repository metadata collected from the GitHub REST API.

Built using **Python, Scikit-Learn, Streamlit, and the GitHub API**.

---

## 🎯 Problem Statement

Can we predict whether a GitHub repository will become successful?

For this project, a repository is considered **successful** if it has:

```text
Stars >= 100
```

The goal is to learn patterns from repository activity, contributors, forks, issues, topics, and age, then predict the probability that a project reaches the 100-star milestone.

---

# 🔍 Why This Dataset Is Different

Many GitHub ML projects accidentally create an easy classification problem.

Example:

```text
Negative class → repos with 0–20 stars
Positive class → repos with 500+ stars
```

A model can separate those almost perfectly.

Instead, this project deliberately focuses on the difficult boundary region around **100 stars**.

### Stratified Star-Bucket Sampling

Repositories are collected from 8 star buckets:

| Bucket    | Range     | Dataset Share |
| --------- | --------- | ------------- |
| 0-star    | 0         | 20%           |
| 1–9       | 1–9       | 10%           |
| 10–49     | 10–49     | 10%           |
| 50–99     | 50–99     | 20%           |
| 100–299   | 100–299   | 20%           |
| 300–999   | 300–999   | 8%            |
| 1000–9999 | 1000–9999 | 7%            |
| 10000+    | 10000+    | 5%            |

The 50–99 and 100–299 buckets are intentionally oversampled because they lie directly around the classification boundary.

This forces the model to learn meaningful patterns rather than simply separating tiny projects from famous ones.

---

# 🌍 Multi-Language Collection

Repositories are collected across multiple ecosystems:

* Python
* JavaScript
* TypeScript
* Go
* Rust
* Java

Queries are distributed across multiple date windows to avoid collecting only recently trending repositories.

---

# 📊 Dataset Features

### Repository Metrics

* Forks
* Open Issues
* Contributors
* Commits
* Repository Size
* Repository Age
* Topics Count
* Wiki Enabled
* Projects Enabled
* Downloads Enabled
* Primary Language

### Engineered Features

```text
commits_per_day
fork_to_issue_ratio
contributors_per_commit
has_topics
```

### Target Variable

```python
success = 1 if stars >= 100 else 0
```

---

# 🧠 Models

Two models are trained and compared:

### Logistic Regression

Advantages:

* Fast
* Interpretable
* Strong baseline

### Random Forest

Advantages:

* Handles nonlinear relationships
* Robust to noisy features
* Provides feature importance scores

The best-performing model is automatically saved.

---

# 📈 Results

Dataset size: **200 repositories**

### Logistic Regression

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.68  |
| Precision | 0.62  |
| Recall    | 0.50  |
| F1 Score  | 0.55  |
| ROC-AUC   | 0.80  |

### Random Forest

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.85  |
| Precision | 0.78  |
| Recall    | 0.88  |
| F1 Score  | 0.82  |
| ROC-AUC   | 0.91  |

The Random Forest model significantly outperformed Logistic Regression and was selected as the final model.

---

# 📌 Most Important Features

Top predictors according to Random Forest:

1. Forks
2. Fork-to-Issue Ratio
3. Open Issues
4. Contributors
5. Topics Count
6. Commits
7. Repository Age

This suggests that community engagement signals are stronger predictors than language choice.

---

# 📂 Project Structure

```text
github-success-predictor/
│
├── app.py
├── collect_data.py
├── train.py
├── predict.py
├── generate_demo_data.py
├── requirements.txt
├── README.md
│
├── data/
│   └── repos.csv
│
├── models/
│   ├── feature_columns.csv
│   ├── model_results.csv
│   └── eda_plots/
│
└── .gitignore
```

---

# ⚙️ Installation

```bash
git clone https://github.com/kumudasrip/github-success-predictor.git

cd github-success-predictor

pip install -r requirements.txt
```

---

# 📥 Collect Real Data

Generate a GitHub Personal Access Token:

https://github.com/settings/tokens

Then run:

```bash
python collect_data.py --token YOUR_GITHUB_TOKEN --output data/repos.csv --total 600
```

---

# 🏋️ Train the Model

```bash
python train.py --data data/repos.csv --output models/
```

This will:

* Perform EDA
* Generate plots
* Train models
* Compare performance
* Save the best model

---

# 🔮 Predict a Repository

CLI:

```bash
python predict.py --url https://github.com/tiangolo/fastapi
```

---

# 🌐 Run the Web App

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

Paste any GitHub repository URL and receive:

* Success prediction
* Success probability
* Feature-based explanation

---

# 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Requests
* Matplotlib
* Seaborn
* Streamlit
* GitHub REST API

---

# 🚀 Future Improvements

* XGBoost / LightGBM
* Time-series growth prediction
* GitHub Actions metrics
* README quality scoring
* NLP analysis of project descriptions
* Contributor network features

---

## Author

Built by Kumuda as a machine learning project exploring GitHub repository growth patterns.