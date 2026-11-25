import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def normalize_target(series: pd.Series) -> pd.Series:
    if series.dtype == object or str(series.dtype).startswith("string"):
        mapped = series.astype(str).str.strip().str.lower().map({
            "1": 1, "true": 1, "yes": 1, "churn": 1, "churned": 1, "y": 1,
            "0": 0, "false": 0, "no": 0, "not churn": 0, "retained": 0, "n": 0,
        })
        if mapped.notna().all():
            return mapped.astype(int)
    return pd.to_numeric(series, errors="raise").astype(int)


def main() -> None:
    df = pd.read_csv("customer_churn_standard_dataset.csv")

    print("=== 1. EXPLORACIÓN INICIAL ===")
    print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores faltantes por columna:")
    print(df.isnull().sum())

    y = normalize_target(df["churn"])
    print("\nDistribución de churn (0 = permanece, 1 = abandona):")
    print(y.value_counts())
    print(f"Proporción de churn: {y.mean():.2%}")

    drop_cols = [c for c in ["churn", "customer_id"] if c in df.columns]
    X = df.drop(columns=drop_cols).copy()

    cat_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    # Exportar dataset procesado con transformador ajustado solo en train (sin fuga de datos)
    preprocessor.fit(X_train, y_train)
    feature_names = preprocessor.get_feature_names_out()
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()

    df_train_out = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)
    df_train_out["churn"] = y_train.values
    df_train_out["split"] = "train"

    df_test_out = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)
    df_test_out["churn"] = y_test.values
    df_test_out["split"] = "test"

    processed_df = pd.concat([df_train_out, df_test_out], axis=0).sort_index()
    if "customer_id" in df.columns:
        processed_df.insert(0, "customer_id", df["customer_id"].values)
    processed_df.to_csv("customer_churn_processed.csv", index=False)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    rows = []
    cms = {}
    fitted_pipes = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rows.append({
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        })
        cms[name] = confusion_matrix(y_test, y_pred)
        fitted_pipes[name] = pipe

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv("model_comparison.csv", index=False)

    print("\n=== 2. COMPARACIÓN DE MODELOS (TEST SET 20%) ===")
    print(metrics_df.to_string(index=False))

    for name, cm in cms.items():
        print(f"\nMatriz de confusión - {name}:")
        print(cm)

    # 1) Gráfico comparativo de métricas
    metrics_melt = metrics_df.set_index("model")[["accuracy", "precision", "recall", "f1_score"]]
    ax = metrics_melt.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Model Comparison on Test Set")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(metrics_melt.index, rotation=0)
    ax.legend(title="Metric", loc="lower right")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.close()

    # 2) Matrices de confusión de los 3 modelos
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax_i, (name, cm) in zip(axes, cms.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn (0)", "Churn (1)"])
        disp.plot(ax=ax_i, cmap="Blues", values_format="d", colorbar=False)
        ax_i.set_title(name)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.close()

    # 3) Importancia de variables del Random Forest
    rf_pipe = fitted_pipes["Random Forest"]
    rf_model = rf_pipe.named_steps["classifier"]
    rf_names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
    imp_series = pd.Series(rf_model.feature_importances_, index=rf_names).sort_values(ascending=True)
    top_imp = imp_series.tail(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    top_imp.plot(kind="barh", ax=ax)
    ax.set_title("Top Feature Importances - Random Forest")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()