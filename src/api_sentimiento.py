"""
API de análisis de sentimiento para Tours PV.

Mejoras:
- Uso de modelos Pydantic para validación.
- Manejo de errores con HTTPException.
- Endpoint /health para revisar estado del modelo.
- Carga de modelo/vectorizador con caché (lru_cache).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple, List

from datetime import datetime
import csv


import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
#  Configuración de rutas
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
UI_DIR = BASE_DIR / "ui"

MODEL_PATH = MODELS_DIR / "modelo_sentimiento.joblib"
VEC_PATH = MODELS_DIR / "vectorizador_tfidf.joblib"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"


# Carpeta para logs / datos generados en producción
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# Archivo donde se guardarán las reseñas analizadas
REVIEWS_LOG_PATH = DATA_DIR / "resenas_analizadas.csv"
RUTA_RESENAS = DATA_DIR / "resenas_analizadas.csv"

# ---------------------------------------------------------------------------
#  Esquemas (Pydantic)
# ---------------------------------------------------------------------------

class SentimentRequest(BaseModel):
    """Entrada de la API para analizar sentimiento."""
    texto: str = Field(
        ...,
        min_length=3,
        description="Texto del comentario a analizar (mínimo 3 caracteres).",
        example="El tour estuvo increíble, la pasé muy bien."
    )


class SentimentResponse(BaseModel):
    """Salida estándar del análisis de sentimiento."""
    sentimiento: str
    probabilidad: float

class Resena(BaseModel):
    timestamp: datetime
    texto: str
    sentimiento: str
    probabilidad: float


class HealthResponse(BaseModel):
    """Respuesta del endpoint /health."""
    status: str
    model_loaded: bool
    vectorizer_loaded: bool


# ---------------------------------------------------------------------------
#  Carga de artefactos de ML
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def cargar_modelos() -> Tuple[Any, Any, Any]:
    if not MODEL_PATH.exists() or not VEC_PATH.exists() or not ENCODER_PATH.exists():
        raise RuntimeError(
            "No se encontró el modelo, el vectorizador o el encoder.\n"
            f"Esperado modelo: {MODEL_PATH}\n"
            f"Esperado vectorizador: {VEC_PATH}\n"
            f"Esperado encoder: {ENCODER_PATH}\n"
            "Ejecuta antes: python -m src.entrenar_modelo"
        )

    try:
        modelo = joblib.load(MODEL_PATH)
        vectorizador = joblib.load(VEC_PATH)
        encoder = joblib.load(ENCODER_PATH)
    except Exception as exc:
        raise RuntimeError(f"Error al cargar los artefactos de ML: {exc}") from exc

    return modelo, vectorizador, encoder


# ---------------------------------------------------------------------------
#  Inicialización de la aplicación FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="API de Análisis de Sentimiento - Tours PV",
    description="Clasificación de comentarios en sentimientos (positivo, negativo, neutro).",
    version="1.0.0",
)

# CORS (por si luego quieres consumir desde otra app web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción conviene restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar la UI estática
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Prueba de conexión")
def raiz() -> dict:
    """Endpoint simple para comprobar que la API está viva."""
    return {
        "mensaje": "API de análisis de sentimiento funcionando",
        "version": "1.0.0",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado de la API y del modelo",
)
def health_check() -> HealthResponse:
    """
    Verifica si el modelo y el vectorizador se cargan correctamente.
    Útil para monitoreo o pruebas rápidas.
    """
    try:
        modelo, vectorizador, encoder = cargar_modelos()
        ok_model = modelo is not None
        ok_vec = vectorizador is not None
    except Exception:
        # No propagamos el error completo aquí, solo indicamos que algo falló.
        ok_model = False
        ok_vec = False

    status_str = "ok" if (ok_model and ok_vec) else "error"

    return HealthResponse(
        status=status_str,
        model_loaded=ok_model,
        vectorizer_loaded=ok_vec,
    )


@app.post(
    "/analizar",
    response_model=SentimentResponse,
    summary="Analizar sentimiento de un texto",
)
def analizar_sentimiento(payload: SentimentRequest) -> SentimentResponse:
    """
    Recibe un texto y devuelve el sentimiento estimado y su probabilidad.
    """
    texto = payload.texto.strip()

    if not texto:
        # Validación extra, aunque Pydantic ya controla longitud mínima
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El texto no puede estar vacío.",
        )

    # Cargar modelo, vectorizador y encoder
    try:
        modelo, vectorizador, encoder = cargar_modelos()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    # 1️⃣ Calcular predicción SOLO UNA VEZ
    try:
        X_vec = vectorizador.transform([texto])

        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X_vec)[0]
            idx = int(proba.argmax())
            # idx es 0,1,2 → lo pasamos por el encoder para obtener la etiqueta original
            sentimiento = str(encoder.classes_[idx])
            prob = float(proba[idx])
        else:
            # Por compatibilidad, aunque LogisticRegression sí tiene predict_proba
            y_pred_enc = modelo.predict(X_vec)[0]
            sentimiento = str(encoder.inverse_transform([y_pred_enc])[0])
            prob = 1.0

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar el texto: {exc}",
        ) from exc

    # 2️⃣ Guardar reseña analizada en CSV (sin volver a calcular nada)
    try:
        guardar_resena(texto, sentimiento, prob)
    except Exception as exc:
        # No queremos que falle el endpoint solo por no poder guardar el log
        print(f"[WARN] No se pudo guardar la reseña: {exc}")

    return SentimentResponse(
        sentimiento=sentimiento,          # ahora será 'negativo', 'neutral' o 'positivo'
        probabilidad=round(prob, 4),
    )
@app.get(
    "/resenas",
    response_model=List[Resena],
    summary="Obtener historial de reseñas analizadas",
)
def obtener_resenas(limit: int = 50) -> List[Resena]:
    """
    Devuelve las últimas 'limit' reseñas analizadas guardadas en CSV.
    """
    try:
        return leer_resenas(limit=limit)
    except Exception as exc:
        print(f"[WARN] No se pudo leer el historial de reseñas: {exc}")
        return []
    
# ---------------------------------------------------------------------------
#  Funciones auxiliares     
# ---------------------------------------------------------------------------



def guardar_resena(texto: str, sentimiento: str, probabilidad: float) -> None:
    """
    Guarda una reseña analizada en un archivo CSV.

    - Crea el archivo con encabezados si no existe.
    - Agrega una línea por cada petición a /analizar.
    """
    is_new_file = not REVIEWS_LOG_PATH.exists()

    with REVIEWS_LOG_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["timestamp", "texto", "sentimiento", "probabilidad"])

        writer.writerow(
            [
                datetime.utcnow().isoformat(timespec="seconds"),
                texto,
                sentimiento,
                f"{probabilidad:.4f}",
            ]
        )

def leer_resenas(limit: int = 50) -> List[Resena]:
    """Lee las reseñas guardadas en el CSV y devuelve las más recientes."""
    if not RUTA_RESENAS.exists():
        return []

    resenas: List[Resena] = []

    with RUTA_RESENAS.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # Esperamos: timestamp, texto, sentimiento, prob
            if len(row) != 4:
                continue

            ts_str, texto, sentimiento, prob_str = row
            try:
                ts = datetime.fromisoformat(ts_str)
                prob = float(prob_str)
            except Exception:
                # Si una línea viene mal, la saltamos
                continue

            resenas.append(
                Resena(
                    timestamp=ts,
                    texto=texto,
                    sentimiento=sentimiento,
                    probabilidad=prob,
                )
            )

    # Ordenar por fecha descendente (más recientes primero)
    resenas.sort(key=lambda r: r.timestamp, reverse=True)
    return resenas[:limit]
