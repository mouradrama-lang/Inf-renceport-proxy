import asyncio, json, os, time
from fastapi import FastAPI, Request from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

# URL du service gratuit InferencePort AI
UPSTREAM = os.environ.get("UPSTREAM_URL", "https://sharktide-lightning.hf.space").rstrip("/")
RETRIES = int(os.environ.get("RETRIES", "3"))
CONNECT_TIMEOUT = float(os.environ.get("CONNECT_TIMEOUT", "8"))
READ_TIMEOUT = float(os.environ.get("READ_TIMEOUT", "600"))
CHAT_READ_TIMEOUT = float(os.environ.get("CHAT_READ_TIMEOUT", "120"))
MODEL_CACHE_TTL = int(os.environ.get("MODEL_CACHE_TTL", "300"))

# Optionnel : un jeton pour protéger ton proxy. Laisse vide pour un accès ouvert.
TOKEN = os.environ.get("PROXY_TOKEN", "")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

app = FastAPI(title="inferenceport-proxy")  
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Cache pour la liste des modèles
model_cache = {"ts": 0.0, "data": None}
models_lock = asyncio.Lock()

def auth_ok(request: Request) -> bool:
    if not TOKEN:
        return True
    h = request.headers.get("Authorization", "")
    return h == f"Bearer {TOKEN}" or h == TOKEN

async def fetch_models():
    async with models_lock:
        if model_cache["data"] and (time.time() - model_cache["ts"]) < MODEL_CACHE_TTL:
            return model_cache["data"]
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20, connect=CONNECT_TIMEOUT), trust_env=False) as c:
                r = await c.get(f"{UPSTREAM}/gen/models", headers={"User-Agent": UA})
                if r.status_code == 200:
                    model_cache["data"] = r.json()
                    model_cache["ts"] = time.time()
                    return model_cache["data"]
        except Exception as e:
            print(f"[models] fetch error: {e}", flush=True)
        return model_cache["data"]

async def upstream_call(path: str, method: str, body: bytes = None, params=None, read_timeout=READ_TIMEOUT):
    url = f"{UPSTREAM}{path}"
    headers = {"User-Agent": UA, "Content-Type": "application/json", "Accept": "*/*"}
    last = None
    for attempt in range(RETRIES):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(read_timeout, connect=CONNECT_TIMEOUT), trust_env=False) as c:
                r = await c.request(method, url, content=body, params=params, headers=headers)
                if r.status_code in (200, 201, 202):
                    return r.status_code, r.headers, r.content
                if r.status_code not in (429, 500, 502, 503, 504):
                    return r.status_code, r.headers, r.content
                last = r
        except Exception as e:
            print(f"[upstream] error: {e}", flush=True)
            last = e
        await asyncio.sleep(1 + attempt * 0.5)
    if isinstance(last, Exception):
        raise last
    return last.status_code, last.headers, last.content

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/v1/healthz")
async def healthz():
    """Vérifie que le proxy est en ligne et que le service amont est joignable."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10, connect=5), trust_env=False) as c:
            r = await c.get(f"{UPSTREAM}/gen/models", headers={"User-Agent": UA})
            upstream_ok = r.status_code == 200
    except Exception:
        upstream_ok = False
    return {"status": "ok", "upstream": upstream_ok}

@app.get("/v1/models")
async def list_models():
    """Liste les modèles disponibles (texte, image, vidéo)."""
    data = await fetch_models()
    if data is None:
        return JSONResponse({"error": "Impossible de récupérer les modèles"}, status_code=502)
    return data

@app.post("/v1/videos")
async def create_video(request: Request):
    """Crée une tâche de génération de vidéo (asynchrone)."""
    if not auth_ok(request):
        return JSONResponse({"error": "Non autorisé"}, status_code=401)
    body = await request.body()
    try:
        status, headers, content = await upstream_call(
            "/gen/video", "POST", body=body, read_timeout=READ_TIMEOUT
        )
    except Exception as e:
        return JSONResponse({"error": f"Erreur amont : {str(e)}"}, status_code=502)
    return JSONResponse(content=json.loads(content), status_code=status)

@app.get("/v1/videos/{job_id}")
async def get_video_status(job_id: str, request: Request):
    """Récupère l'état d'une tâche vidéo."""
    if not auth_ok(request):
        return JSONResponse({"error": "Non autorisé"}, status_code=401)
    try:
        status, headers, content = await upstream_call(
            f"/gen/video/{job_id}", "GET"
        )
    except Exception as e:
        return JSONResponse({"error": f"Erreur amont : {str(e)}"}, status_code=502)
    return JSONResponse(content=json.loads(content), status_code=status)

@app.get("/v1/videos/{job_id}/content")
async def get_video_content(job_id: str, request: Request):
    """Télécharge le fichier MP4 d'une vidéo terminée."""
    if not auth_ok(request):
        return JSONResponse({"error": "Non autorisé"}, status_code=401)
    try:
        status, headers, content = await upstream_call(
            f"/gen/video/{job_id}/content", "GET", read_timeout=READ_TIMEOUT
        )
    except Exception as e:
        return JSONResponse({"error": f"Erreur amont : {str(e)}"}, status_code=502)
    media_type = headers.get("content-type", "video/mp4")
    return StreamingResponse(iter([content]), media_type=media_type)

# ═══════════════════════════════════════════════════════════════
# LANCEMENT (pour le développement local)
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
