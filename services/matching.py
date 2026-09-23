import logging

from bson import ObjectId

from db import utc_now
from services.face_service import FaceAnalysisError


logger = logging.getLogger(__name__)


class MatchingService:
    def __init__(self, database, face_service, email_service, threshold: float):
        self.db = database
        self.face = face_service
        self.email = email_service
        self.threshold = threshold

    def _possible_matches(self, embedding, missing_documents):
        try:
            normalized_embedding = self.face.normalize_embedding(embedding, "incoming")
        except FaceAnalysisError as exc:
            logger.error("Cannot match invalid incoming embedding: %s", exc)
            return []
        matches = []
        for missing in missing_documents:
            try:
                distance = self.face.cosine_distance(normalized_embedding, missing["faceEmbedding"])
            except (FaceAnalysisError, ValueError) as exc:
                logger.error(
                    "Skipping malformed stored embedding collection=missing_persons id=%s: %s",
                    missing.get("_id"),
                    exc,
                )
                continue
            if distance <= self.threshold:
                matches.append((missing, distance))
        return sorted(matches, key=lambda item: item[1])

    def match_found_person(self, found_id, embedding):
        found = self.db.found.find_one({"_id": ObjectId(found_id)})
        active_missing = self.db.missing.find({"status": "missing", "submissionStatus": "completed", "matchingExcluded": {"$ne": True}, "faceEmbedding": {"$exists": True}})
        possible = self._possible_matches(embedding, active_missing)
        if not possible:
            self.db.found.update_one({"_id": found["_id"]}, {"$set": {"matchStatus": "no_match", "updatedAt": utc_now()}})
            return self._notify_no_match(found)
        for missing, distance in possible:
            self._create_match(missing, found, distance)
            body = self._match_body(missing, found, distance)
            self.email.send_safely([missing["email"], found["email"]], "Possible Match Found", body)
        self.db.found.update_one({"_id": found["_id"]}, {"$set": {"matchStatus": "possible_match", "updatedAt": utc_now()}})
        return len(possible)

    def match_missing_person(self, missing_id, embedding):
        missing = self.db.missing.find_one({"_id": ObjectId(missing_id)})
        eligible_found = self.db.found.find({"submissionStatus": "completed", "matchingExcluded": {"$ne": True}, "matchStatus": {"$in": ["no_match", "pending"]}, "faceEmbedding": {"$exists": True}})
        possible = self._possible_matches(embedding, eligible_found)
        for found, distance in possible:
            self._create_match(missing, found, distance)
            body = self._match_body(missing, found, distance)
            self.email.send_safely([missing["email"], found["email"]], "Possible Match Found", body)
            self.db.found.update_one({"_id": found["_id"]}, {"$set": {"matchStatus": "possible_match", "updatedAt": utc_now()}})
        return len(possible)

    def _create_match(self, missing, found, distance):
        self.db.matches.update_one(
            {"missingPersonId": missing["_id"], "foundPersonId": found["_id"], "verificationStatus": {"$nin": ["confirmed", "rejected"]}},
            {"$setOnInsert": {
                "missingPersonId": missing["_id"], "foundPersonId": found["_id"],
                "distance": distance, "similarity": 1 - distance,
                "model": self.face.model_name, "threshold": self.threshold,
                "status": "possible_match", "verificationStatus": "unverified", "createdAt": utc_now(),
            }},
            upsert=True,
        )

    def _notify_no_match(self, found):
        return self.email.send_safely(
            [found["email"]],
            "No Match Found",
            f"Found Person: {found.get('name', 'Name not provided')}\n\nNo match found currently.\nYour data has been saved and will be checked with future entries.\n\nAI matching is not a guarantee; a future entry may identify a possible match.",
        )

    def _match_body(self, missing, found, distance):
        similarity = self.face.similarity_percent(distance)
        return (
            "A possible match has been detected.\n\n"
            f"Missing Person: {missing['name']}\n"
            f"Found Person: {found.get('name', 'Name not provided')}\n"
            f"Found Location: {found['foundLocation']}\n"
            f"Confidence: {similarity}% (cosine similarity representation, not a probability)\n\n"
            f"Please contact: {missing['contactNumber']}\n\n"
            "This is only a possible AI-assisted match. Human verification is required before confirming the person's identity."
        )
