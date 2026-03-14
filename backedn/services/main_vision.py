import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.vision_routes import router

app = FastAPI(title="QA Bot — Vision Service")

_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
_extra = [o.strip() for o in _origins_env.split(",") if o.strip()]
ALLOWED_ORIGINS = list(dict.fromkeys([
    "http://localhost:3000",
    "http://localhost:5173",
    "https://ragchatbot.sharadsnaik.in",
] + _extra))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Vision Service is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 6002))
    uvicorn.run(app, host="0.0.0.0", port=port)
