# Student GPA Predictor

A machine learning project that predicts student GPA (0.0–4.0) from behavioral and demographic features, built for a Machine Learning course group project.

## Model

**Architecture: CNN-BiGRU**

```
Conv1D → MaxPooling1D → Bidirectional GRU → Dense(ReLU) → Dropout → Dense(linear)
```

The input features are treated as a 1D sequence, allowing the convolutional layer to extract local feature patterns before the bidirectional GRU captures dependencies across features.

Hyperparameters were tuned using **Keras Tuner RandomSearch** (20 trials), optimizing for validation MAE.

## Dataset

- **Source**: Student performance dataset
- **Size**: 2,392 students
- **Features (10)**: Ethnicity, Parental Education, Gender, Weekly Study Time, Absences, Tutoring, Parental Support, Extracurricular, Sports, Music
- **Target**: GPA (continuous, 0.0–4.0)
- **Preprocessing**: MinMaxScaler normalization

Features like StudentID, Age, Volunteering, and GradeClass were dropped after correlation analysis.

## Performance

| Metric | Value |
|---|---|
| Mean Absolute Error (MAE) | 0.184 |
| Mean Squared Error (MSE) | 0.054 |
| R² Score | 0.934 |
| Accuracy within ±0.25 GPA | 72.0% |

## Files

| File | Description |
|---|---|
| `ml_project.ipynb` | Full training pipeline: EDA, preprocessing, tuning, evaluation |
| `app.py` | Streamlit web app for interactive GPA prediction |
| `student_gpa_model.keras` | Saved trained model |
| `minmax_scaler.pkl` | Saved MinMaxScaler (must match training preprocessing) |

## Running Locally

> Tested on Windows 11. Requires Python 3.10+.

```powershell
# 1. Clone the repo
git clone https://github.com/pembasmikuman/Deployed-ML-project.git
cd Deployed-ML-project

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install streamlit tensorflow pandas numpy scikit-learn joblib

# 4. Run the app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.
