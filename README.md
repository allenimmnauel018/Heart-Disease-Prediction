# Heart Disease Prediction using Machine Learning

This project demonstrates how to build, train, and evaluate machine learning models for **heart disease prediction** using the **UCI Heart Disease Dataset**. The notebook includes **data preprocessing, exploratory data analysis (EDA), model training, evaluation, and predictions on new user data**.

---

## Project Structure

* `heart_disease_uci.csv` → Original dataset.
* `heart_dataset.csv` → Example user dataset for predictions.
* `Heart_user_template.csv` → Generated template for user input.
* `heart_rf_model.pkl` → Trained Random Forest model.
* `heart_scaler.pkl` → StandardScaler object for feature scaling.
* `notebook.ipynb` → Main Jupyter Notebook (analysis, training, predictions).

---

## Workflow

### 1. Data Loading & Cleaning

* Handle missing values in numeric and categorical features.
* Convert target (`num`) into binary classification: **0 = No Disease, 1 = Disease**.

### 2. Exploratory Data Analysis (EDA)

* Histograms of numeric features.
* Correlation heatmap of features.

### 3. Feature Engineering

* One-hot encoding for categorical variables.
* Standardization using `StandardScaler`.

### 4. Model Training

* **Logistic Regression** → Baseline model.
* **Random Forest Classifier** → Improved performance & feature importance.

### 5. Model Evaluation

* Accuracy, Classification Report, Confusion Matrix.
* Feature importance visualization for Random Forest.

### 6. Model Deployment

* Save trained model (`joblib`).
* Save scaler for consistent preprocessing.
* Generate **user input template CSV** for predictions.

### 7. Predictions on New Data

* Load new dataset (`heart_dataset.csv`).
* Apply preprocessing pipeline.
* Predict heart disease risk.
* Append predictions to user data.

---

## Results

* Logistic Regression: Moderate accuracy, interpretable.
* Random Forest: Higher accuracy, useful feature importance insights.
* Predictions stored in user dataset with column **`Heart_Disease_Prediction`**.

---

## How to Run

1. Clone the repository and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Open the Jupyter Notebook:

   ```bash
   jupyter notebook notebook.ipynb
   ```

3. Train the model, save it, and generate predictions.

4. Use `Heart_user_template.csv` as input format for new predictions.

---

## 🧑‍⚕️ Example Output

Sample prediction results:

| age | sex | cp | trestbps | chol | ... | Heart\_Disease\_Prediction |
| --- | --- | -- | -------- | ---- | --- | -------------------------- |
| 63  | 1   | 3  | 145      | 233  | ... | 1                          |

**1 → Disease, 0 → No Disease**

---

## 📌 Future Improvements

* Hyperparameter tuning with GridSearchCV.
* Add Deep Learning (ANN) for comparison.
* Deploy as a Flask/FastAPI web app.

---

## 📖 References

* [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
* Scikit-learn Documentation
