# 🎣 Fish Catch Prediction System — Sri Lanka

> **4th Year Machine Learning Project | Vavuniya Campus of the University Of Jaffna — Faculty of Technological Studies**  


---

## 📌 Project Overview

This project develops an end-to-end **machine learning system** to predict fish catch amounts for the Sri Lankan fishing industry. Sri Lanka's fishing sector employs over 560,000 people and contributes ~1.3% of national GDP. Yet fishermen often venture to sea without accurate knowledge of expected catch volumes — leading to wasted fuel, insufficient storage, and economic losses.

This system uses environmental conditions, geographic parameters, and operational factors to predict expected fish catch (in kg), empowering fishermen to make smarter, data-driven decisions.

---

## 🎯 Key Achievements

| Metric | Value |
|--------|-------|
| Best Model | LightGBM (Optimized with Optuna) |
| Test R² Score | **62.49%** |
| Mean Absolute Error | **±137.09 kg** |
| Dataset Size | 50,000 records |
| Regions Covered | 10 major Sri Lankan coastal regions |
| Models Compared | 7 algorithms |

---

## 🗂️ Project Structure

```
project/
│
├── app.py                              # Streamlit web application
├── Fish_Catch_Final_Structured.ipynb   # Full ML pipeline (EDA → Training → Evaluation)
├── sri_lanka_fishing_dataset_50k.csv   # Dataset (50,000 fishing records)
├── requirements.txt                    # Python dependencies
├── ml_report.pdf                       # Full technical report
│
└── models/
    ├── fish_catch_model.pkl            # Trained LightGBM model
    ├── scaler.pkl                      # StandardScaler object
    ├── feature_columns.pkl             # Feature column list
    └── model_metadata.pkl              # Model metrics & metadata
```

---

## 🌊 Dataset Features

The dataset covers **10 coastal regions**: Negombo, Puttalam, Kalutara, Galle, Matara, Hambantota, Trincomalee, Batticaloa, Kalmunai, Jaffna

| Category | Features |
|----------|----------|
| 📍 Geographic | Region, Latitude, Longitude, Distance from Shore, Water Depth, Fishing Zone |
| 🌤️ Environmental | Sea Surface Temperature, Chlorophyll, Wind Speed, Wave Height |
| 📅 Temporal | Season, Moon Phase, Time of Day, Date |
| 🚤 Operational | Boat Type, Fish Species, Fisherman Experience |
| 🎯 Target | `Catch_kg` (fish catch in kilograms) |

---

## 🤖 Machine Learning Pipeline

1. **Data Preprocessing** — Missing value imputation (median by group), feature engineering
2. **Feature Engineering** — Distance-Depth Ratio, Wind-Wave Ratio, Is_Monsoon flag, temporal features
3. **Encoding** — One-Hot Encoding for all 7 categorical variables (~50 features after encoding)
4. **Train/Test Split** — 75% train / 25% test (random_state=42)
5. **Scaling** — StandardScaler (fitted on train only, to prevent data leakage)
6. **Model Training** — 7 algorithms compared (Linear Regression, Lasso, Ridge, Decision Tree, Random Forest, Gradient Boosting, LightGBM)
7. **Evaluation** — R², MAE, MSE, RMSE

### Model Comparison Results

| Model | Train R² | Test R² | MAE (kg) |
|-------|----------|---------|----------|
| **LightGBM** ✅ | 0.6769 | **0.6257** | **136.99** |
| CatBoost | 0.7017 | 0.6220 | 137.45 |
| Gradient Boosting | 0.8070 | 0.6174 | 138.57 |
| XGBoost | 0.9288 | 0.6104 | 139.93 |
| Random Forest | 0.9209 | 0.5782 | 144.14 |
| Linear Regression | 0.5612 | 0.5584 | 143.07 |

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fish-catch-prediction-sri-lanka.git
cd fish-catch-prediction-sri-lanka
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model (if models/ folder is missing)

Open and run `Fish_Catch_Final_Structured.ipynb` in Jupyter Notebook, or run:

```bash
jupyter notebook Fish_Catch_Final_Structured.ipynb
```

### 4. Launch the Web Application

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🖥️ Web Application Features

Built with **Streamlit 1.26.0**, the app has 3 tabs:

- **🎣 Predict Catch** — Input fishing trip details and get instant prediction with recommendations
- **📊 Model Performance** — View R², MAE, RMSE metrics and regional statistics
- **📖 User Guide** — Documentation for fishermen and non-technical users

**Input parameters include:** Region, season, sea temperature, chlorophyll level, wind speed, wave height, moon phase, boat type, target species, and fisherman experience.

**Output includes:** Predicted catch (kg), confidence range, condition rating (Excellent/Good/Fair/Poor), fuel & ice recommendations, and an interactive gauge chart.

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| ML Library | scikit-learn, LightGBM |
| Web Framework | Streamlit 1.26.0 |
| Visualization | Plotly 5.16.1, Matplotlib, Seaborn |
| Data Processing | Pandas 2.0.3, NumPy 1.24.3 |
| Model Persistence | Pickle |
| Notebook | Jupyter |

---

## 📊 Results Summary

- **Best Model:** LightGBM with Optuna hyperparameter tuning
- **Test R²:** 0.6249 — model explains 62.5% of variance in fish catch
- **MAE:** ±137 kg — practical accuracy for real-world trip planning
- **Generalization gap:** Train R² − Test R² = 0.051 (good, no significant overfitting)

---

## ⚠️ Limitations

- Dataset is **synthetic** (generated based on realistic Sri Lankan fishing statistics) — real-world validation with actual catch records is needed
- R² of ~62% leaves room for improvement; fish catch is inherently variable
- No real-time integration with oceanographic/weather APIs (yet)
- Single model for all fish species; species-specific models may improve accuracy

---

## 🔮 Future Work

- Integrate real catch data from Sri Lanka Department of Fisheries
- Add real-time weather/SST API integration
- Develop mobile app (Flutter/React Native) for offline use at sea
- Multi-language support: Sinhala 🇱🇰, Tamil, English
- Deep learning models (LSTM for time-series patterns)

---

## 📄 License

This project was developed as part of the **4th Year Machine Learning coursework** at the University of Vavuniya, Faculty of Technological Studies. For academic use only.

---

## 👨‍💻 Author

**K.M.P.S. Gunasekara**   
Faculty of Technological Studies, Vavuniya Campus of the University of jaffna  
Date: February 2026
