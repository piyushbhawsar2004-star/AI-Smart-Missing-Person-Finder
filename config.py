import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-development")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/missing_person_finder")
    EMAIL_USER = os.getenv("EMAIL_USER", "")
    EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
    FACE_MODEL = os.getenv("FACE_MODEL", "Facenet512")
    FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.35"))
    UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", str(BASE_DIR / "uploads")))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "8")) * 1024 * 1024
    SESSION_COOKIE_SECURE = _as_bool(os.getenv("SESSION_COOKIE_SECURE"), False)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    ADMIN_SESSION_HOURS = int(os.getenv("ADMIN_SESSION_HOURS", "8"))
    OWNER_NAME = "Piyush Bhawsar"
    OWNER_EMAIL = "piyushbhawsar05@gmail.com"
    OWNER_PHONE = "7697725054"

    @classmethod
    def init_app(cls, app):
        app.config.from_object(cls)
        app.config["UPLOAD_FOLDER"] = Path(app.config["UPLOAD_FOLDER"])
        cls.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        (cls.UPLOAD_FOLDER / "missing").mkdir(exist_ok=True)
        (cls.UPLOAD_FOLDER / "found").mkdir(exist_ok=True)
        app.config.from_object(cls)
