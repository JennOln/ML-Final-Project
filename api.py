import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import jax.numpy as jnp
import numpy as np
import uvicorn

app = FastAPI()

# CONFIGURACIÓN CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# CARGAR CEREBRO DE LA RED NEURONAL (MLP)
def load_mlp_weights():
    try:
        weights = jnp.load("final_weights.npz")
        return [weights[f"arr_{i}"] for i in range(len(weights.files))]
    except Exception as e:
        print(f"Aviso: No se encontraron los pesos ({e}). Corre Metaflow primero.")
        return None

mlp_params = load_mlp_weights()

# CARGAR ESCALADOR PARA EL ORÁCULO
try:
    with open("scaler_params.json", "r") as f:
        scaler = json.load(f)
except Exception as e:
    print(f"Aviso: No se encontró scaler_params.json ({e}).")
    scaler = None

# FUNCIONES MATEMÁTICAS (JAX)
def relu(x): return jnp.maximum(0, x)
def sigmoid(z): return 1 / (1 + jnp.exp(-z))

def mlp_predict(params, X):
    W1, b1, W2, b2, W3, b3 = params
    a1 = relu(jnp.dot(X, W1) + b1)
    a2 = relu(jnp.dot(a1, W2) + b2)
    return sigmoid(jnp.dot(a2, W3) + b3)


# --- ESTRUCTURA DE DATOS PYDANTIC ---
class LibroInput(BaseModel):
    stars: float
    price: float
    genre: str
    is_established_author: bool
    ku: bool
    pick: bool
    choice: bool


# Rutas de la API (BACKEND) 
@app.post("/api/predict")
async def predict(libro: LibroInput):
    try:
        if mlp_params is None or scaler is None:
            return JSONResponse(
                content={"error": "Faltan los pesos o el escalador. Corre pipeline.py run primero."}, 
                status_code=500
            )

        data = libro.dict()
        
        # Escalar (Normalizar) las variables numéricas exactamente como en el entrenamiento
        stars_scaled = (float(data['stars']) - scaler['mu_stars']) / (scaler['sigma_stars'] + 1e-8)
        price_scaled = (float(data['price']) - scaler['mu_price']) / (scaler['sigma_price'] + 1e-8)
        
        features = [stars_scaled, price_scaled]
        
        # Macro-Géneros 
        macro_genres = [
            'Culture', 'Education', 'Entertainment', 'Health', 
            'Nonfiction', 'Others', 'Romance', 'Science Fiction & Fantasy', 
            'Teen & Young Adult', 'Thriller', 'childs'
        ]
        for g in macro_genres:
            features.append(1.0 if data['genre'] == f"genre_{g}" else 0.0)

        # Variables Booleanas
        features.extend([
            1.0 if data.get('is_established_author', False) else 0.0,
            1.0 if data.get('ku', False) else 0.0,
            1.0 if data.get('pick', False) else 0.0,
            1.0 if data.get('choice', False) else 0.0
        ])

        # Inferencia con JAX
        X_input = jnp.array([features])
        salida_red = mlp_predict(mlp_params, X_input)
        
        # FIX: Extraemos el número exacto de la matriz usando .item()
        probabilidad = float(salida_red.item())
        
        # Determinar si es élite 
        es_elite = probabilidad > 0.42 
        
        return {
            "probabilidad": probabilidad,
            "es_elite": es_elite,
            "mensaje": "¡Será un Éxito! ✨" if es_elite else "Libro Estándar 📚"
        }

    # MICRÓFONO DE ERRORES (Atrapa cualquier falla matemática o de matrices)
    except Exception as e:
        import traceback
        print("\n=== CHOQUE EN EL SERVIDOR ===")
        print(traceback.format_exc())
        
        return JSONResponse(
            content={"error": f"Error interno del modelo: {str(e)}"},
            status_code=500
        )

@app.get("/api/all-metrics")
async def get_metrics():
    # Leemos las métricas que Metaflow dejó 
    try:
        with open("metrics_cache.json", "r") as f:
            cache_metrics = json.load(f)
        return JSONResponse(content=cache_metrics)
    except FileNotFoundError:
        return JSONResponse(content={"error": "Corre Metaflow primero para generar las métricas."})


# RUTAS FRONTEND (STATIC FILES) ---
@app.get("/")
async def serve_home():
    return FileResponse("index.html")

app.mount("/resources", StaticFiles(directory="resources"), name="resources")
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)