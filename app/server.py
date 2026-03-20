import os
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
PORT = int(os.environ.get("PORT", "8420"))
HOST = os.environ.get("HOST", "127.0.0.1")

if __package__ in {None, ""}:
    sys.path.insert(0, str(APP_ROOT.parent))

from app.routes import register_routes


def create_app() -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "connect-src 'self' https://basemaps.cartocdn.com https://*.basemaps.cartocdn.com; "
            "img-src 'self' data: blob: https://basemaps.cartocdn.com https://*.basemaps.cartocdn.com; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; "
            "worker-src 'self' blob:; "
            "base-uri 'none'; "
            "frame-ancestors 'none'; "
            "form-action 'self'"
        )
        return response

    app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC_ROOT / "index.html")

    register_routes(app)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    print(f"\n  MoveBench Trajectory Viewer: http://{HOST}:{PORT}\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
