import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

from bson import ObjectId
from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

from config import Config
from db import Database, public_document, utc_now
from services.email_service import EmailService
from services.face_service import FaceAnalysisError, FaceService
from services.matching import MatchingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
PHONE_RE = re.compile(r"^\+?[0-9 ()-]{7,20}$")


def create_app(test_config=None):
    app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent / "frontend"), static_url_path="/static")
    Config.init_app(app)
    if test_config:
        app.config.update(test_config)
    app.config["UPLOAD_FOLDER"] = Path(app.config["UPLOAD_FOLDER"])
    upload_root = app.config["UPLOAD_FOLDER"]
    upload_root.mkdir(parents=True, exist_ok=True)
    (upload_root / "missing").mkdir(exist_ok=True)
    (upload_root / "found").mkdir(exist_ok=True)
    CORS(app, resources={r"/api/*": {"origins": app.config.get("CORS_ORIGINS", "*")}})

    try:
        database = Database(app.config["MONGO_URI"])
        database.ensure_indexes()
    except Exception:
        logger.exception("MongoDB is unavailable during startup")
        database = None

    face_service = FaceService(app.config["FACE_MODEL"])
    email_service = EmailService(app.config["EMAIL_USER"], app.config["EMAIL_APP_PASSWORD"])
    matcher = MatchingService(database, face_service, email_service, app.config["FACE_MATCH_THRESHOLD"]) if database else None
    app.extensions["database"] = database
    app.extensions["face_service"] = face_service
    app.extensions["email_service"] = email_service
    app.extensions["matcher"] = matcher
    logger.info("Application source=%s face_model=%s", Path(__file__).resolve(), app.config["FACE_MODEL"])

    def db_or_error():
        if database is None:
            raise RuntimeError("Database is unavailable")
        return database

    def admin_required(handler):
        @wraps(handler)
        def wrapped(*args, **kwargs):
            if not session.get("admin_id"):
                return jsonify({"error": "Authentication required"}), 401
            if session.get("expires_at", 0) < datetime.now(timezone.utc).timestamp():
                session.clear()
                return jsonify({"error": "Session expired"}), 401
            return handler(*args, **kwargs)
        return wrapped

    def validate_common(form, fields):
        missing = [field for field in fields if not str(form.get(field, "")).strip()]
        if missing:
            return f"Missing required fields: {', '.join(missing)}"
        email = str(form.get("email", "")).strip()
        phone = str(form.get("contactNumber", "")).strip()
        if email and not EMAIL_RE.match(email):
            return "Please enter a valid email address"
        if phone and not PHONE_RE.match(phone):
            return "Please enter a valid contact number"
        return None

    def save_image(upload: FileStorage, category: str):
        if not upload or not upload.filename:
            raise ValueError("An image is required")
        original = secure_filename(upload.filename)
        suffix = Path(original).suffix.lower().lstrip(".")
        if suffix not in ALLOWED_EXTENSIONS or not upload.mimetype.startswith("image/"):
            raise ValueError("Please upload a valid image file")
        target = app.config["UPLOAD_FOLDER"] / category / f"{uuid.uuid4().hex}.{suffix}"
        try:
            upload.stream.seek(0)
            with Image.open(upload.stream) as image:
                image.verify()
            upload.stream.seek(0)
            with Image.open(upload.stream) as image:
                if image.width < 80 or image.height < 80:
                    raise ValueError("Please upload a larger, clearer image")
                image.load()
            upload.stream.seek(0)
            target.write_bytes(upload.stream.read())
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Please upload a valid image file") from exc
        return target.relative_to(app.config["UPLOAD_FOLDER"]).as_posix()

    def load_image_path(relative_path):
        root = app.config["UPLOAD_FOLDER"].resolve()
        candidate = (root / relative_path).resolve()
        if root not in candidate.parents:
            raise ValueError("Invalid image path")
        return candidate

    def form_datetime(value):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Please enter a valid date and time") from exc

    @app.get("/")
    def home():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/health")
    def health():
        try:
            db_or_error().ping()
            return jsonify({"status": "ok", "database": "ok"})
        except Exception:
            return jsonify({"status": "degraded", "database": "unavailable"}), 503

    @app.post("/api/missing")
    def create_missing():
        required = ["fullName", "age", "gender", "lastSeenLocation", "dateTime", "clothesDescription", "email", "contactNumber"]
        error = validate_common(request.form, required)
        if error:
            return jsonify({"error": error}), 400
        image_path = None
        try:
            age = int(request.form["age"])
            if age < 0 or age > 130:
                raise ValueError
            date_time = form_datetime(request.form["dateTime"])
            image_path = save_image(request.files.get("photo"), "missing")
            absolute_path = str(load_image_path(image_path))
            embedding = face_service.embedding(absolute_path)
            database = db_or_error()
            now = utc_now()
            document = {
                "name": request.form["fullName"].strip(), "age": age, "gender": request.form["gender"].strip(),
                "imagePath": image_path, "lastSeenLocation": request.form["lastSeenLocation"].strip(),
                "dateTime": date_time, "clothesDescription": request.form["clothesDescription"].strip(),
                "identificationMarks": request.form.get("identificationMarks", "").strip(),
                "email": request.form["email"].strip(), "contactNumber": request.form["contactNumber"].strip(),
                "faceEmbedding": embedding, "embeddingModel": app.config["FACE_MODEL"], "embeddingDimension": len(embedding),
                "status": "missing", "submissionStatus": "completed",
                "notificationStatus": "pending", "createdAt": now, "updatedAt": now,
            }
            result = database.missing.insert_one(document)
            document["_id"] = result.inserted_id
            email_ok = email_service.send_safely([document["email"]], "Missing Person Report Submitted", "Your missing person report has been successfully registered.")
            if not email_ok:
                database.audit_event("email_failure", "system", {"type": "missing_submission", "recordId": str(result.inserted_id)})
            database.missing.update_one({"_id": result.inserted_id}, {"$set": {"notificationStatus": "sent" if email_ok else "failed", "updatedAt": utc_now()}})
            if matcher:
                matcher.match_missing_person(str(result.inserted_id), embedding)
            return jsonify({"message": "Submitted successfully. Check your email."}), 201
        except FaceAnalysisError as exc:
            if image_path:
                load_image_path(image_path).unlink(missing_ok=True)
            logger.warning("Missing face analysis rejected submission: %s", exc)
            return jsonify({"error": str(exc)}), 422
        except (ValueError, TypeError) as exc:
            if image_path:
                load_image_path(image_path).unlink(missing_ok=True)
            return jsonify({"error": str(exc)}), 400
        except Exception:
            if image_path:
                load_image_path(image_path).unlink(missing_ok=True)
            logger.exception("Missing person submission failed")
            return jsonify({"error": "We could not save this report. Please try again."}), 500

    @app.post("/api/found")
    def create_found():
        required = ["fullName", "foundLocation", "dateTime", "genderApprox", "ageApprox", "clothesDescription", "email", "contactNumber"]
        error = validate_common(request.form, required)
        if error:
            return jsonify({"error": error}), 400
        image_path = None
        try:
            age = int(request.form["ageApprox"])
            if age < 0 or age > 130:
                raise ValueError("Please enter a valid approximate age")
            date_time = form_datetime(request.form["dateTime"])
            image_path = save_image(request.files.get("photo"), "found")
            embedding = face_service.embedding(str(load_image_path(image_path)))
            database = db_or_error()
            now = utc_now()
            document = {
                "name": request.form["fullName"].strip(), "imagePath": image_path, "foundLocation": request.form["foundLocation"].strip(), "dateTime": date_time,
                "genderApprox": request.form["genderApprox"].strip(), "ageApprox": age,
                "clothesDescription": request.form["clothesDescription"].strip(), "email": request.form["email"].strip(),
                "contactNumber": request.form["contactNumber"].strip(), "faceEmbedding": embedding,
                "embeddingModel": app.config["FACE_MODEL"], "embeddingDimension": len(embedding),
                "matchStatus": "pending", "verificationStatus": "unverified", "submissionStatus": "completed",
                "notificationStatus": "pending", "createdAt": now, "updatedAt": now,
            }
            result = database.found.insert_one(document)
            document["_id"] = result.inserted_id
            matching_completed = matcher is None
            if matcher:
                try:
                    matcher.match_found_person(str(result.inserted_id), embedding)
                    matching_completed = True
                except Exception:
                    database.audit_event("matching_failure", "system", {"type": "found_submission", "recordId": str(result.inserted_id)})
                    logger.exception("Found report saved but matching failed for record=%s", result.inserted_id)
            database.found.update_one({"_id": result.inserted_id}, {"$set": {"notificationStatus": "sent" if matching_completed else "failed", "updatedAt": utc_now()}})
            return jsonify({"message": "Result has been sent to your email"}), 201
        except FaceAnalysisError as exc:
            if image_path:
                load_image_path(image_path).unlink(missing_ok=True)
            logger.warning("Found face analysis rejected submission: %s", exc)
            return jsonify({"error": str(exc)}), 422
        except (ValueError, TypeError) as exc:
            if image_path:
                load_image_path(image_path).unlink(missing_ok=True)
            return jsonify({"error": str(exc)}), 400
        except Exception:
            if image_path:
                load_image_path(image_path).unlink(missing_ok=True)
            logger.exception("Found person submission failed")
            return jsonify({"error": "We could not process this capture. Please try again."}), 500

    @app.post("/api/auth/login")
    def login():
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        try:
            user = db_or_error().users.find_one({"username": username, "active": True})
            if not user or not check_password_hash(user["passwordHash"], password):
                return jsonify({"error": "Invalid username or password"}), 401
            session.clear()
            session["admin_id"] = str(user["_id"])
            session["username"] = username
            session["expires_at"] = (datetime.now(timezone.utc) + timedelta(hours=app.config["ADMIN_SESSION_HOURS"])).timestamp()
            db_or_error().audit_event("login", username)
            return jsonify({"message": "Signed in"})
        except Exception:
            logger.exception("Login failed")
            return jsonify({"error": "Authentication is temporarily unavailable"}), 503

    @app.post("/api/auth/logout")
    def logout():
        session.clear()
        return jsonify({"message": "Signed out"})

    @app.get("/api/admin/me")
    @admin_required
    def admin_me():
        return jsonify({"username": session["username"]})

    @app.get("/api/admin/missing")
    @admin_required
    def admin_missing():
        documents = db_or_error().missing.find({"submissionStatus": "completed"}, {"faceEmbedding": 0, "imagePath": 0}).sort("createdAt", -1)
        return jsonify({"items": [public_document(document) for document in documents]})

    @app.get("/api/admin/found")
    @admin_required
    def admin_found():
        documents = db_or_error().found.find({"$or": [{"submissionStatus": "completed"}, {"submissionStatus": {"$exists": False}, "faceEmbedding": {"$exists": True}, "imagePath": {"$exists": True}}]}, {"faceEmbedding": 0, "imagePath": 0}).sort("createdAt", -1)
        return jsonify({"items": [public_document(document) for document in documents]})

    @app.get("/api/admin/matches")
    @admin_required
    def admin_matches():
        database = db_or_error()
        items = []
        for match in database.matches.find().sort("createdAt", -1):
            missing = database.missing.find_one({"_id": match["missingPersonId"], "submissionStatus": "completed"}, {"faceEmbedding": 0, "imagePath": 0})
            found = database.found.find_one({"_id": match["foundPersonId"], "submissionStatus": "completed"}, {"faceEmbedding": 0, "imagePath": 0})
            item = public_document(match)
            item["missing"] = public_document(missing) if missing else None
            item["found"] = public_document(found) if found else None
            items.append(item)
        return jsonify({"items": items})

    @app.get("/api/admin/history")
    @admin_required
    def admin_history():
        database = db_or_error()
        items = []
        for document in database.missing.find({"status": "found", "submissionStatus": "completed"}, {"faceEmbedding": 0, "imagePath": 0}).sort("foundAt", -1):
            item = public_document(document)
            found = database.found.find_one({"_id": document.get("matchedFoundPersonId"), "$or": [{"submissionStatus": "completed"}, {"submissionStatus": {"$exists": False}, "faceEmbedding": {"$exists": True}, "imagePath": {"$exists": True}}]}, {"faceEmbedding": 0, "imagePath": 0})
            item["foundPerson"] = public_document(found) if found else None
            items.append(item)
        return jsonify({"items": items})

    @app.get("/api/admin/audit-logs")
    @admin_required
    def admin_audit():
        documents = db_or_error().audit.find().sort("createdAt", -1).limit(200)
        return jsonify({"items": [public_document(document) for document in documents]})

    @app.post("/api/admin/matches/<match_id>/confirm")
    @admin_required
    def confirm_match(match_id):
        database = db_or_error()
        try:
            match = database.matches.find_one({"_id": ObjectId(match_id)})
        except Exception:
            match = None
        if not match or match.get("verificationStatus") in {"confirmed", "rejected"}:
            return jsonify({"error": "Match is not available for confirmation"}), 404
        now = utc_now()
        database.matches.update_one({"_id": match["_id"]}, {"$set": {"verificationStatus": "confirmed", "status": "confirmed", "verifiedAt": now, "verifiedBy": session["username"]}})
        database.missing.update_one({"_id": match["missingPersonId"]}, {"$set": {"status": "found", "foundAt": now, "confirmedAt": now, "matchedFoundPersonId": match["foundPersonId"], "updatedAt": now}})
        database.found.update_one({"_id": match["foundPersonId"]}, {"$set": {"matchStatus": "resolved", "verificationStatus": "confirmed", "updatedAt": now}})
        database.audit_event("confirm_match", session["username"], {"matchId": match_id})
        return jsonify({"message": "Possible match confirmed"})

    @app.post("/api/admin/matches/<match_id>/reject")
    @admin_required
    def reject_match(match_id):
        database = db_or_error()
        payload = request.get_json(silent=True) or {}
        reason = str(payload.get("reason", "")).strip()[:500]
        try:
            match = database.matches.find_one({"_id": ObjectId(match_id)})
        except Exception:
            match = None
        if not match or match.get("verificationStatus") in {"confirmed", "rejected"}:
            return jsonify({"error": "Match is not available for rejection"}), 404
        now = utc_now()
        database.matches.update_one({"_id": match["_id"]}, {"$set": {"verificationStatus": "rejected", "status": "rejected", "rejectionReason": reason, "verifiedAt": now, "verifiedBy": session["username"]}})
        database.found.update_one({"_id": match["foundPersonId"]}, {"$set": {"matchStatus": "pending", "verificationStatus": "rejected", "updatedAt": now}})
        database.audit_event("reject_match", session["username"], {"matchId": match_id, "reason": reason})
        return jsonify({"message": "Possible match rejected"})

    @app.get("/api/admin/images/<category>/<path:image_path>")
    @admin_required
    def protected_image(category, image_path):
        if category not in {"missing", "found"}:
            return jsonify({"error": "Not found"}), 404
        try:
            path = load_image_path(f"{category}/{image_path}")
            return send_from_directory(path.parent, path.name)
        except ValueError:
            return jsonify({"error": "Not found"}), 404

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "Image is too large. Please use a smaller image."}), 413

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found"}), 404
        return send_from_directory(app.static_folder, "index.html")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
