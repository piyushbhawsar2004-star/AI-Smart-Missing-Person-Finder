import argparse
import logging

from app import app
from services.face_service import EXPECTED_EMBEDDING_DIMENSION, FaceAnalysisError, FaceService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_collection(collection, collection_name, quarantine):
    face = FaceService(app.config["FACE_MODEL"])
    malformed = 0
    for document in collection.find({}, {"faceEmbedding": 1}):
        try:
            face.normalize_embedding(document.get("faceEmbedding"), collection_name)
        except FaceAnalysisError as exc:
            malformed += 1
            logger.error("Malformed embedding collection=%s id=%s: %s", collection_name, document["_id"], exc)
            if quarantine:
                collection.update_one(
                    {"_id": document["_id"]},
                    {"$set": {"embeddingStatus": "malformed", "matchingExcluded": True}},
                )
    logger.info("collection=%s malformed=%d expected_dimension=%d", collection_name, malformed, EXPECTED_EMBEDDING_DIMENSION)
    return malformed


def main():
    parser = argparse.ArgumentParser(description="Inspect or quarantine malformed stored face embeddings.")
    parser.add_argument("--quarantine", action="store_true", help="Mark malformed records as excluded from matching.")
    args = parser.parse_args()
    database = app.extensions["database"]
    missing = inspect_collection(database.missing, "missing_persons", args.quarantine)
    found = inspect_collection(database.found, "found_persons", args.quarantine)
    raise SystemExit(1 if missing or found else 0)


if __name__ == "__main__":
    main()