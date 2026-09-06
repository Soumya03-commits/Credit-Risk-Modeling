from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model_bundle.joblib"
BACKEND_PATH = ROOT / "trained_features.csv"

CATEGORICAL_COLUMNS = [
    "MARITALSTATUS",
    "EDUCATION",
    "GENDER",
    "last_prod_enq2",
    "first_prod_enq2",
]
MANUAL_NUMERIC_COLUMNS = ["AGE", "NETMONTHLYINCOME"]

EDUCATION_MAPPING = {
    "SSC": 1,
    "12TH": 2,
    "GRADUATE": 3,
    "UNDER GRADUATE": 3,
    "POST-GRADUATE": 4,
    "OTHERS": 1,
    "PROFESSIONAL": 3,
}


def load_and_prepare_data() -> pd.DataFrame:
    case_one = pd.read_excel(ROOT / "case_study1.xlsx")
    case_two = pd.read_excel(ROOT / "case_study2.xlsx")

    case_one = case_one.loc[case_one["Age_Oldest_TL"] != -99999].copy()

    removable = [
        column
        for column in case_two.columns
        if (case_two[column] == -99999).sum() > 10000
    ]
    case_two = case_two.drop(columns=removable)
    for column in case_two.columns:
        case_two = case_two.loc[case_two[column] != -99999]

    data = case_one.merge(case_two, on="PROSPECTID", how="inner")
    return data


def select_features(data: pd.DataFrame) -> list[str]:
    numeric_columns = [
        column
        for column in data.columns
        if data[column].dtype != "object" and column != "PROSPECTID"
    ]

    selected_numeric = []
    target = data["Approved_Flag"]
    for column in numeric_columns:
        groups = [
            data.loc[target == label, column]
            for label in ["P1", "P2", "P3", "P4"]
        ]
        _, p_value = f_oneway(*groups)
        if p_value <= 0.05:
            selected_numeric.append(column)

    vif_data = data[selected_numeric].copy()
    selected_after_vif = []
    while vif_data.shape[1] > 0:
        values = vif_data.to_numpy(dtype=float)
        inverse = np.linalg.pinv(values.T @ values)
        vif_values = np.diag(inverse) * np.diag(values.T @ values)
        highest_index = int(np.nanargmax(vif_values))
        if vif_values[highest_index] <= 5:
            selected_after_vif.extend(vif_data.columns.tolist())
            break
        vif_data = vif_data.drop(columns=[vif_data.columns[highest_index]])

    selected_after_vif = list(dict.fromkeys(
        selected_after_vif + [
            column for column in MANUAL_NUMERIC_COLUMNS
            if column in data.columns
        ]
    ))
    return selected_after_vif + CATEGORICAL_COLUMNS


def main() -> None:
    data = load_and_prepare_data()
    selected_features = select_features(data)
    data = data[selected_features + ["PROSPECTID", "Approved_Flag"]].copy()
    data["EDUCATION"] = data["EDUCATION"].map(EDUCATION_MAPPING)

    encoded = pd.get_dummies(data, columns=CATEGORICAL_COLUMNS, dtype=int)
    target = encoded.pop("Approved_Flag")
    prospect_ids = encoded.pop("PROSPECTID")

    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    model.fit(encoded, target_encoded)

    backend = encoded.copy()
    backend.insert(0, "PROSPECTID", prospect_ids)
    backend.to_csv(BACKEND_PATH, index=False)

    joblib.dump(
        {
            "model": model,
            "trained_columns": encoded.columns.tolist(),
            "label_encoder": label_encoder,
            "education_mapping": EDUCATION_MAPPING,
            "categorical_columns": CATEGORICAL_COLUMNS,
        },
        MODEL_PATH,
    )

    print(f"Saved model bundle to {MODEL_PATH}")
    print(f"Saved backend feature data to {BACKEND_PATH}")
    print(f"Training features: {len(encoded.columns)}")


if __name__ == "__main__":
    main()