"""Train and evaluate a beginner-friendly predictive-maintenance model."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
IMAGE_DIR = ROOT / "images"
MODEL_DIR = ROOT / "models"

TARGET = "Machine failure"
FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_FEATURES = FEATURES[1:]


def load_and_clean_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the data and remove exact duplicate rows, if any."""
    data = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    return data.drop_duplicates().copy()


def create_eda_charts(data: pd.DataFrame) -> None:
    """Create a small set of charts useful during the internship defence."""
    IMAGE_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=data, x=TARGET, hue=TARGET, palette="Set2", legend=False)
    ax.set(title="Machine Failure Class Distribution", xlabel="Machine failure (0 = No, 1 = Yes)")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "class_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(data[NUMERIC_FEATURES + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Between Sensor Variables")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "correlation_heatmap.png", dpi=180)
    plt.close()

    sample = data.sample(n=min(3000, len(data)), random_state=42)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=sample,
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        hue=TARGET,
        palette={0: "#4c78a8", 1: "#e45756"},
        alpha=0.65,
    )
    plt.title("Torque and Rotational Speed by Machine Condition")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "torque_vs_speed.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=data, x=TARGET, y="Tool wear [min]", hue=TARGET, palette="Set2", legend=False)
    plt.title("Tool Wear for Normal and Failed Machines")
    plt.xlabel("Machine failure (0 = No, 1 = Yes)")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "tool_wear_by_failure.png", dpi=180)
    plt.close()


def specificity_score(y_true, y_pred) -> float:
    """Return TN / (TN + FP)."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)


def build_pipeline() -> Pipeline:
    """Encode product type, then fit a deliberately small decision tree."""
    preprocessing = ColumnTransformer(
        [("type", OneHotEncoder(handle_unknown="ignore"), ["Type"])],
        remainder="passthrough",
    )
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def evaluate_model(data: pd.DataFrame):
    """Use a stratified hold-out test and stratified five-fold cross-validation."""
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    metrics = {
        "test_rows": len(y_test),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "accuracy": accuracy_score(y_test, predictions),
        "sensitivity_recall": recall_score(y_test, predictions),
        "specificity": specificity_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        build_pipeline(),
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "recall": "recall",
            "specificity": make_scorer(specificity_score),
            "precision": "precision",
            "f1": "f1",
        },
    )
    metrics["cross_validation"] = {
        name.removeprefix("test_"): {
            "mean": float(values.mean()), "standard_deviation": float(values.std())
        }
        for name, values in cv_results.items()
        if name.startswith("test_")
    }

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Normal", "Failure"],
        cmap="Blues",
        colorbar=False,
    )
    plt.title("Decision Tree Confusion Matrix")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    encoder = pipeline.named_steps["preprocessing"]
    feature_names = encoder.get_feature_names_out()
    importance = pd.Series(
        pipeline.named_steps["model"].feature_importances_, index=feature_names
    ).sort_values()
    importance.plot.barh(figsize=(8, 5), color="#4c78a8")
    plt.title("Decision Tree Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "feature_importance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(20, 10))
    plot_tree(
        pipeline.named_steps["model"],
        feature_names=feature_names,
        class_names=["Normal", "Failure"],
        filled=True,
        rounded=True,
        fontsize=7,
        max_depth=3,
    )
    plt.title("Simplified View of the Decision Tree (First Four Levels)")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "decision_tree.png", dpi=180)
    plt.close()

    return pipeline, metrics


def predict_machine_condition(pipeline: Pipeline, sensor_readings: dict) -> dict:
    """Predict one machine condition and return the probability of failure."""
    row = pd.DataFrame([sensor_readings], columns=FEATURES)
    prediction = int(pipeline.predict(row)[0])
    probability = float(pipeline.predict_proba(row)[0, 1])
    return {
        "prediction": "Failure risk" if prediction else "Normal",
        "failure_probability": probability,
    }


def main() -> None:
    data = load_and_clean_data()
    print(f"Rows: {len(data):,} | Columns: {data.shape[1]}")
    print(f"Missing values: {int(data.isna().sum().sum())}")
    print(f"Failure counts:\n{data[TARGET].value_counts().sort_index()}")

    create_eda_charts(data)
    pipeline, metrics = evaluate_model(data)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "decision_tree_pipeline.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nTest-set results")
    for name, value in metrics.items():
        if name != "cross_validation":
            print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
    print("\nFive-fold cross-validation")
    for name, values in metrics["cross_validation"].items():
        print(f"{name}: {values['mean']:.4f} ± {values['standard_deviation']:.4f}")

    example = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1350,
        "Torque [Nm]": 60.0,
        "Tool wear [min]": 210,
    }
    print("\nExample prediction:", predict_machine_condition(pipeline, example))


if __name__ == "__main__":
    main()
