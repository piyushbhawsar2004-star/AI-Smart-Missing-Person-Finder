import logging

import numpy as np


logger = logging.getLogger(__name__)
EXPECTED_EMBEDDING_DIMENSION = 512


class FaceAnalysisError(Exception):
    pass


class FaceService:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embedding(self, image_path: str):
        try:
            from deepface import DeepFace
        except ImportError as exc:
            raise FaceAnalysisError("Face analysis dependencies are not installed") from exc

        try:
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend="opencv",
                enforce_detection=True,
                align=True,
            )
            if len(faces) != 1:
                raise FaceAnalysisError("Please upload an image containing exactly one clear face")
            representation = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                detector_backend="opencv",
                enforce_detection=True,
                align=True,
            )
            if len(representation) != 1 or not representation[0].get("embedding"):
                raise FaceAnalysisError("A clear face could not be embedded")
            return self.normalize_embedding(representation[0]["embedding"], "generated").tolist()
        except FaceAnalysisError:
            raise
        except Exception as exc:
            logger.exception("DeepFace embedding generation failed for model=%s", self.model_name)
            raise FaceAnalysisError("Face embedding generation failed. Please try another clear image.") from exc

    @staticmethod
    def normalize_embedding(value, label):
        try:
            embedding = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise FaceAnalysisError(f"Invalid {label} face embedding") from exc
        logger.info("Face embedding shape label=%s shape=%s dtype=%s", label, embedding.shape, embedding.dtype)
        if embedding.shape != (EXPECTED_EMBEDDING_DIMENSION,):
            raise FaceAnalysisError(
                f"Expected Facenet512 embedding for {label}, got {embedding.shape}"
            )
        if not np.isfinite(embedding).all():
            raise FaceAnalysisError(f"Invalid numeric values in {label} face embedding")
        return embedding

    @classmethod
    def cosine_distance(cls, left, right):
        found_array = cls.normalize_embedding(left, "found")
        missing_array = cls.normalize_embedding(right, "missing")
        if found_array.shape != missing_array.shape:
            raise FaceAnalysisError(
                f"Embedding dimension mismatch: found={found_array.shape}, missing={missing_array.shape}"
            )
        denominator = np.linalg.norm(found_array) * np.linalg.norm(missing_array)
        if denominator == 0:
            raise FaceAnalysisError("Invalid face embedding")
        distance = 1 - np.dot(found_array, missing_array) / denominator
        return float(np.clip(distance, 0.0, 2.0))

    @staticmethod
    def similarity_percent(distance: float) -> float:
        """Cosine similarity representation, not a probability or identity claim."""
        return round(max(0.0, min(1.0, 1.0 - distance)) * 100, 1)
