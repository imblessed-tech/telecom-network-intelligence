from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from api.routers import health, fault_detection, congestion, anomaly
from api.dependencies import get_artifact_store

app = FastAPI(title="Telecom Network Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

app.include_router(health.router, tags=["health"])
app.include_router(fault_detection.router, tags=["fault-detection"])
app.include_router(congestion.router, tags=["congestion"])
app.include_router(anomaly.router, tags=["anomaly"])


@app.on_event("startup")
def load_artifacts_on_startup():
    # Forces artifact loading at boot rather than on first request
    get_artifact_store()
