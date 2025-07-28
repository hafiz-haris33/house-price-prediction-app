# 🏡 House Price Prediction - California Housing

This project predicts the **median house value** in different districts of California using **machine learning techniques**. It uses the **California Housing dataset**, performs data preprocessing, feature engineering, model selection, evaluation, and deployment as a **Streamlit web application**.

---

## 📊 Dataset

- **Source**: [California Housing Dataset](https://github.com/ageron/data/raw/main/housing.tgz)
- **Attributes**:
  - longitude, latitude  
  - housing_median_age  
  - total_rooms, total_bedrooms  
  - population, households  
  - median_income  
  - median_house_value (target)
  - ocean_proximity

---

## 🧠 Machine Learning Workflow

### 🔹 Step 1: Data Collection
- The housing dataset was programmatically downloaded from an online GitHub source, extracted using the tarfile module, and loaded into a pandas DataFrame.

### 🔹 Step 2: Train-Test Split (Early)
- Split the data using **stratified sampling** based on income category to avoid data leakage.
- Binned median income into categories for stratified sampling.

### 🔹 Step 3: Exploratory Data Analysis (EDA)
- Checked dataset shape, null values, data types, and distribution.
- Visualized:
  - Correlation matrix  
  - Geographic plots (latitude vs longitude)  
  - Histogram plots for numeric features  
  - Scatter plots to observe relationships with `median_house_value`

### 🔹 Step 4: Feature Engineering
- Created new features:
  - `rooms_per_house`  
  - `bedrooms_ratio`  
  - `people_per_house`

### 🔹 Step 5: Data Preprocessing
- Used `Pipeline` and `ColumnTransformer` from `sklearn`:
  - Separate pipelines for:
    - Numerical features  
    - Categorical features (e.g., one-hot encoding)  
    - Derived ratio features
    - Skewed features (Log pipeline)
  - Custom transformer for clustering similarity.

### 🔹 Step 6: Model Selection & Training
Trained multiple models:
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- HistGradientBoosting Regressor  
- XGBoost Regressor  

### 🔹 Step 7: Model Evaluation
- Used **Root Mean Squared Error (RMSE)** for evaluation  
- Performed **cross-validation** with `cross_val_score`  
- Visualized model performance comparison

### 🔹 Step 8: Hyperparameter Tuning
- Tuned top 3 models using:
  - `GridSearchCV`  
  - `RandomizedSearchCV`
- Best model: **Random Forest Regressor (tuned using Random Search)**

### 🔹 Step 9: Final Model Testing
- Evaluated final model on **test set** (previously untouched)  
- RMSE on test set showed excellent generalization  
- Saved model using `joblib`

---

## 🚀 Streamlit Web App

### Features:
- Interactive UI built using **Streamlit**  
- Allows user to input housing attributes  
- Predicts median house value instantly  

---

## ⚙️ Tech Stack

- **Python 3.10+**  
- **Pandas, NumPy, Scikit-learn**  
- **XGBoost, Matplotlib, Seaborn**  
- **Streamlit (App Interface)**  
- **Joblib (Model saving/loading)**  

---

## Download the model
- You can download the model from [Drive](https://drive.google.com/file/d/1wG4g5gF9HIWH49o6Ht53HzDn1I8zyxd0/view?usp=drive_link)

---

## 🖥️ How to Run the App Locally

```bash
# 1. Clone this repository
git clone https://github.com/hafiz-haris33/house-price-prediction-app
cd house-price-prediction

# 2. Create a virtual environment
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Run Streamlit App
streamlit run app.py
```

---

## 🌐 Deployment

App can be deployed on **Streamlit Cloud**:

- Push your code to GitHub  
- Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)  
- Connect your GitHub repo  
- Select `app.py` as the main file and deploy

---

## 👨‍💻 Author

**Hafiz Muhammad Haris Attique**  
_Machine Learning | Data Science Enthusiast_  
📫 [LinkedIn Profile](https://www.linkedin.com/in/hafiz-muhammad-haris-305211361)

---

## ⭐ Support

If you liked this project, feel free to **fork** and **star** this repo.  
Happy coding! 🚀
