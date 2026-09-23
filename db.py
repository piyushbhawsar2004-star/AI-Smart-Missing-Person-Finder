from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError


def utc_now():
    return datetime.now(timezone.utc)


class Database:
    def __init__(self, uri: str):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=2500)
        self.db = self.client.get_default_database()
        self.missing = self.db.missing_persons
        self.found = self.db.found_persons
        self.matches = self.db.match_history
        self.users = self.db.users
        self.audit = self.db.audit_logs

    def ensure_indexes(self):
        self.missing.create_index([("status", ASCENDING), ("createdAt", DESCENDING)])
        self.found.create_index([("matchStatus", ASCENDING), ("createdAt", DESCENDING)])
        self.matches.create_index([("verificationStatus", ASCENDING), ("createdAt", DESCENDING)])
        self.users.create_index("username", unique=True)

    def ping(self):
        self.client.admin.command("ping")

    def audit_event(self, action, actor, details=None):
        self.audit.insert_one({
            "action": action,
            "actor": actor,
            "details": details or {},
            "createdAt": utc_now(),
        })


def public_document(document):
    result = dict(document)
    result.pop("faceEmbedding", None)
    result.pop("imagePath", None)
    result["id"] = str(result.pop("_id"))
    for key in ("createdAt", "updatedAt", "dateTime", "foundAt", "confirmedAt", "verifiedAt"):
        if result.get(key):
            result[key] = result[key].isoformat()
    for key in ("missingPersonId", "foundPersonId", "verifiedBy", "matchedFoundPersonId"):
        if result.get(key):
            result[key] = str(result[key])
    return result
