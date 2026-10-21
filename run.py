import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent  # backend/
load_dotenv(BASE_DIR / ".env")


if __name__ == "__main__":
    is_dev = os.getenv("APP_ENV", "development") == "development"

    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8080")),
        reload=is_dev,
        reload_dirs=[str(BASE_DIR / "src")] if is_dev else None,
        app_dir=str(BASE_DIR / "src"),
    )