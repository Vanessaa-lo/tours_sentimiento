"""
Script para entrenar el modelo de análisis de sentimiento.

Versión: LogisticRegression (Regresión Logística) multiclase.
- Usa TF-IDF para vectorizar texto.
- Codifica las etiquetas (positivo, neutral, negativo) con LabelEncoder.
- Calcula métricas (accuracy, F1) y guarda un classification_report.
- Guarda modelo, vectorizador, encoder y métricas en la carpeta /models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
#  RUTAS
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "modelo_sentimiento.joblib"
VEC_PATH = MODELS_DIR / "vectorizador_tfidf.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

DATA_DIR = BASE_DIR / "data"
RUTA_DATOS = DATA_DIR / "reseñas.csv"  # 👈 tu archivo real


# ---------------------------------------------------------------------------
#  CARGA Y LIMPIEZA DE DATOS
# ---------------------------------------------------------------------------

def cargar_datos(
    ruta: Path,
    texto_col: str = "texto",
    target_col: str = "sentimiento",
) -> Tuple[pd.Series, pd.Series]:
    """
    Carga el dataset desde un CSV y devuelve X (textos) e y (etiquetas).

    - Normaliza la columna de sentimiento:
      * pasa a minúsculas
      * elimina espacios al inicio/fin
      * convierte "nan" (texto) y cadenas vacías en NaN reales
      * elimina filas sin sentimiento
      * elimina clases con menos de 2 ejemplos
    """

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {ruta}\n"
            "Verifica la ruta o ajusta RUTA_DATOS en entrenar_modelo.py"
        )

    df = pd.read_csv(ruta)

    if texto_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"Las columnas esperadas '{texto_col}' y/o '{target_col}' "
            f"no existen en el CSV. Columnas disponibles: {list(df.columns)}"
        )

    # Aseguramos tipo string en texto
    df[texto_col] = df[texto_col].astype(str)

    # Normalizar sentimientos
    df[target_col] = (
        df[target_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Tratar "nan" literal y vacío como valores faltantes
    df[target_col] = df[target_col].replace({"nan": pd.NA, "": pd.NA})

    # Eliminar filas sin sentimiento definido
    df = df.dropna(subset=[target_col])

    print("Distribución de clases tras limpieza:")
    print(df[target_col].value_counts())

    # Eliminar clases con menos de 2 ejemplos
    counts = df[target_col].value_counts()
    clases_validas = counts[counts >= 2].index
    df = df[df[target_col].isin(clases_validas)]

    print("Distribución de clases tras filtrar clases con < 2 ejemplos:")
    print(df[target_col].value_counts())

    X = df[texto_col]
    y = df[target_col]

    return X, y


# ---------------------------------------------------------------------------
#  ENTRENAMIENTO DEL MODELO (LOGISTIC REGRESSION)
# ---------------------------------------------------------------------------

def entrenar_modelo(
    X_train: pd.Series,
    y_train_enc,
) -> tuple[LogisticRegression, TfidfVectorizer]:
    """
    Entrena el vectorizador TF-IDF y un modelo de Regresión Logística.

    - TF-IDF con uni- y bigramas.
    - Regresión Logística multiclase (one-vs-rest o multinomial según solver).
    """

    # Vectorizador TF-IDF
    vectorizador = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
    )

    X_train_vec = vectorizador.fit_transform(X_train)

    modelo = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        multi_class="multinomial",
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )


    modelo.fit(X_train_vec, y_train_enc)

    return modelo, vectorizador


# ---------------------------------------------------------------------------
#  EVALUACIÓN DEL MODELO
# ---------------------------------------------------------------------------

def evaluar_modelo(
    modelo: LogisticRegression,
    vectorizador: TfidfVectorizer,
    X_test: pd.Series,
    y_test_enc,
    encoder: LabelEncoder,
) -> dict:
    """
    Evalúa el modelo y devuelve un diccionario de métricas.
    """

    X_test_vec = vectorizador.transform(X_test)
    y_pred_enc = modelo.predict(X_test_vec)

    acc = accuracy_score(y_test_enc, y_pred_enc)
    f1 = f1_score(y_test_enc, y_pred_enc, average="weighted")

    target_names = list(encoder.classes_)

    report = classification_report(
        y_test_enc,
        y_pred_enc,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "classes": target_names,
        "classification_report": report,
    }

    return metrics


# ---------------------------------------------------------------------------
#  GUARDAR MODELO, VECTORIZADOR, ENCODER Y MÉTRICAS
# ---------------------------------------------------------------------------

def guardar_artefactos(
    modelo: LogisticRegression,
    vectorizador: TfidfVectorizer,
    encoder: LabelEncoder,
    metrics: dict,
) -> None:
    """
    Guarda el modelo, el vectorizador, el encoder y las métricas en disco.
    """

    joblib.dump(modelo, MODEL_PATH)
    joblib.dump(vectorizador, VEC_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Vectorizador guardado en: {VEC_PATH}")
    print(f"Encoder guardado en: {ENCODER_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("Cargando datos...")
    X, y = cargar_datos(RUTA_DATOS, texto_col="texto", target_col="sentimiento")

    print(f"Total de muestras después de limpieza: {len(X)}")

    value_counts = y.value_counts()
    print("Distribución final de clases:")
    print(value_counts)

    min_class_count = value_counts.min()
    use_stratify = min_class_count >= 2

    if not use_stratify:
        print(
            "⚠ Advertencia: al menos una clase tiene menos de 2 ejemplos. "
            "Se hará la división sin 'stratify'."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if use_stratify else None,
    )

    # Codificar etiquetas con LabelEncoder
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    print("Clases del encoder:", list(encoder.classes_))

    print(f"Entrenando modelo (Regresión Logística)... (train: {len(X_train)}, test: {len(X_test)})")
    modelo, vectorizador = entrenar_modelo(X_train, y_train_enc)

    print("Evaluando modelo...")
    metrics = evaluar_modelo(modelo, vectorizador, X_test, y_test_enc, encoder)

    print("Resultados principales:")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")

    print("Guardando artefactos...")
    guardar_artefactos(modelo, vectorizador, encoder, metrics)

    print("Entrenamiento finalizado correctamente ✅")


if __name__ == "__main__":
    main()
