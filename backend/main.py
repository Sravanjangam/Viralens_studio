from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.detect_router import router as detect_router
from backend.routers.caption_router import router as caption_router
from backend.routers.trend_router import router as trend_router
from backend.routers.virality_router import router as virality_router
from backend.routers.full_router import router as full_router
from backend.routers.chat_router import router as chat_router
from backend.image_gen.image_router import router as image_router


app = FastAPI(title="ViraLens API", version="1.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later: lock to domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(detect_router, prefix="/detect", tags=["Detection"])
app.include_router(caption_router, prefix="/caption", tags=["Caption"])
app.include_router(trend_router, prefix="/trend", tags=["Trend"])
app.include_router(virality_router, prefix="/virality", tags=["Virality"])
app.include_router(full_router, prefix="/full", tags=["Full Analysis"])
app.include_router(chat_router, prefix="/chatbot")
app.include_router(image_router)


@app.get("/")
async def root():
    return {"message": "ViraLens API Running 🚀"}
