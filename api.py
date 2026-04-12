import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/all-metrics")
async def get_metrics():
    # Leemos las métricas que Metaflow dejó listas
    try:
        with open("metrics_cache.json", "r") as f:
            cache_metrics = json.load(f)
        return JSONResponse(content=cache_metrics)
    except FileNotFoundError:
        return JSONResponse(content={"error": "Corre Metaflow primero para generar las métricas."})

@app.get("/")
async def serve_home():
    return FileResponse("index.html")

app.mount("/resources", StaticFiles(directory="resources"), name="resources")
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)