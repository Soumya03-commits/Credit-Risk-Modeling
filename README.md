# Credit Risk Streamlit Prototype

## Build the model and backend CSV

From this folder, run:

```powershell
python build_model_artifacts.py
```

This creates:

- `model_bundle.joblib`: trained XGBoost model, label encoder, and feature schema.
- `trained_features.csv`: backend lookup data containing `PROSPECTID` and only trained features.

## Start the application

```powershell
streamlit run app.py
```

The app asks for Prospect ID, marital status, age, education, and net monthly income. It retrieves the remaining trained features from `trained_features.csv`, overrides the five user-entered values, aligns the columns to the saved model schema, and predicts `Approved_Flag`.