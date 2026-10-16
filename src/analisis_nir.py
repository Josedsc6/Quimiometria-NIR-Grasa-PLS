"""Análisis reproducible del dataset NIR Tecator para predecir grasa."""
from pathlib import Path
import urllib.request
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
DATA_URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/modeldata/meats.csv"
DATA_PATH = DATA_DIR / "meats.csv"
RANDOM_STATE = 42


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print("Descargando dataset Tecator desde Rdatasets...")
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0 portfolio-chemometrics"})
        with urllib.request.urlopen(request, timeout=30) as response, open(DATA_PATH, "wb") as out:
            out.write(response.read())
    return pd.read_csv(DATA_PATH)


def savgol_first_derivative(X):
    """Primera derivada Savitzky–Golay aplicada fila a fila a los espectros."""
    return savgol_filter(np.asarray(X), window_length=11, polyorder=2, deriv=1, axis=1)


def metric_row(name, y_true, y_pred, cv_rmse=None, best_params=None):
    return {
        "modelo": name,
        "rmse_test": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_test": float(mean_absolute_error(y_true, y_pred)),
        "r2_test": float(r2_score(y_true, y_pred)),
        "rmse_cv_entrenamiento": float(cv_rmse) if cv_rmse is not None else np.nan,
        "mejores_parametros": str(best_params) if best_params is not None else "",
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = download_dataset()
    spectral_cols = [c for c in df_raw.columns if c.startswith("x_")]
    target_col = "fat"
    if len(spectral_cols) != 100 or target_col not in df_raw.columns:
        raise ValueError("Estructura inesperada del CSV: se esperaban 100 columnas x_ y la variable fat.")

    required = spectral_cols + ["fat", "water", "protein"]
    df = df_raw[required].apply(pd.to_numeric, errors="coerce")
    missing_rows = int(df.isna().any(axis=1).sum())
    df = df.dropna().reset_index(drop=True)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=required).reset_index(drop=True)
    duplicates_removed = before_dedup - len(df)

    X = df[spectral_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    wavelengths = np.linspace(850, 1050, len(spectral_cols))

    # Exploración espectral
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx in np.linspace(0, len(X) - 1, min(20, len(X)), dtype=int):
        ax.plot(wavelengths, X[idx], alpha=0.55, linewidth=1)
    ax.set(title="Espectros NIR de muestras Tecator", xlabel="Longitud de onda (nm)", ylabel="Absorbancia")
    fig.tight_layout(); fig.savefig(FIG_DIR / "01_espectros_nir.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(y, bins=18, edgecolor="black", alpha=0.8)
    ax.set(title="Distribución del contenido de grasa", xlabel="Grasa (%)", ylabel="Número de muestras")
    fig.tight_layout(); fig.savefig(FIG_DIR / "02_distribucion_grasa.png", dpi=160); plt.close(fig)

    # PCA centrado sobre los espectros; se usa con fines exploratorios.
    X_centered = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    scores = pca.fit_transform(X_centered)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=y, cmap="viridis", edgecolor="black", linewidth=0.25)
    ax.set(title="PCA de espectros NIR (color = grasa)", xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)", ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
    fig.colorbar(scatter, ax=ax, label="Grasa (%)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "03_pca_espectros.png", dpi=160); plt.close(fig)

    # Partición solo después de eliminar duplicados exactos.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    max_components = min(15, X_train.shape[0] - 1, X_train.shape[1])
    component_grid = list(range(1, max_components + 1))

    models = {
        "PLS_espectro_original": (
            Pipeline([("pls", PLSRegression(scale=True, max_iter=1000))]),
            {"pls__n_components": component_grid},
        ),
        "PLS_primera_derivada": (
            Pipeline([
                ("derivada", FunctionTransformer(savgol_first_derivative, validate=False)),
                ("pls", PLSRegression(scale=True, max_iter=1000)),
            ]),
            {"pls__n_components": component_grid},
        ),
        "Ridge": (
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
            {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [200], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 4]},
        ),
    }

    metrics, prediction_frames = [], []
    fitted = {}
    for name, (estimator, param_grid) in models.items():
        print(f"Ajustando {name}...")
        search = GridSearchCV(
            estimator, param_grid, scoring="neg_root_mean_squared_error", cv=cv,
            n_jobs=-1, refit=True, error_score="raise"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)
        pred = np.asarray(search.predict(X_test)).reshape(-1)
        cv_rmse = -float(search.best_score_)
        metrics.append(metric_row(name, y_test, pred, cv_rmse, search.best_params_))
        prediction_frames.append(pd.DataFrame({"modelo": name, "grasa_real_pct": y_test, "grasa_predicha_pct": pred, "residuo_pct": y_test - pred}))
        fitted[name] = (search.best_estimator_, pred)

    metrics_df = pd.DataFrame(metrics).sort_values("rmse_test").reset_index(drop=True)
    metrics_df.to_csv(RESULTS_DIR / "metricas_modelos.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(RESULTS_DIR / "predicciones_test.csv", index=False)

    # Gráficas del mejor modelo por RMSE en test. El test no se usa para ajustar.
    best_name = metrics_df.loc[0, "modelo"]
    best_model, best_pred = fitted[best_name]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, best_pred, alpha=0.8, edgecolor="black", linewidth=0.3)
    low = min(float(y_test.min()), float(best_pred.min()))
    high = max(float(y_test.max()), float(best_pred.max()))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set(title=f"Observado vs. predicho — {best_name}", xlabel="Grasa observada (%)", ylabel="Grasa predicha (%)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "04_observado_vs_predicho.png", dpi=160); plt.close(fig)

    residuals = y_test - best_pred
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(best_pred, residuals, alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.axhline(0, linestyle="--")
    ax.set(title=f"Residuos — {best_name}", xlabel="Grasa predicha (%)", ylabel="Residuo (observado − predicho, puntos porcentuales)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "05_residuos.png", dpi=160); plt.close(fig)

    summary = (
        "RESUMEN DE EJECUCIÓN — TECATOR NIR\n"
        f"Filas válidas antes de deduplicar: {before_dedup}\n"
        f"Filas con valores ausentes eliminadas: {missing_rows}\n"
        f"Duplicados exactos eliminados: {duplicates_removed}\n"
        f"Muestras únicas analizadas: {len(df)}\n"
        f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}\n"
        f"Mejor RMSE de prueba: {best_name}\n"
        f"RMSE test: {metrics_df.loc[0, 'rmse_test']:.3f} puntos porcentuales\n"
        f"MAE test: {metrics_df.loc[0, 'mae_test']:.3f} puntos porcentuales\n"
        f"R² test: {metrics_df.loc[0, 'r2_test']:.3f}\n\n"
        "Interpretar estos resultados en el contexto del tamaño de muestra y de la partición utilizada.\n"
        "No equivalen a validación externa industrial ni sustituyen un método de referencia validado.\n"
    )
    (RESULTS_DIR / "resumen_ejecucion.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Resultados guardados en: {RESULTS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
