import io
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

from app import create_app
from services.face_service import FaceAnalysisError, FaceService
from services.matching import MatchingService


class FakeCollection:
    def insert_one(self, document):
        class Result:
            inserted_id = 'test-id'
        return Result()


class FakeDatabase:
    missing = FakeCollection()
    found = FakeCollection()

    def ensure_indexes(self): pass
    def ping(self): pass
    def audit_event(self, *args, **kwargs): pass


def image_bytes():
    output = io.BytesIO()
    Image.new('RGB', (120, 120), 'white').save(output, format='PNG')
    output.seek(0)
    return output


@pytest.fixture
def client(monkeypatch, tmp_path):
    app = create_app({'TESTING': True, 'UPLOAD_FOLDER': Path(tmp_path), 'SECRET_KEY': 'test'})
    fake = FakeDatabase()
    app.extensions['database'] = fake
    app.extensions['matcher'] = None
    app.extensions['face_service'].embedding = lambda path: [1.0, 0.0, 0.0]
    return app.test_client()


def test_health(client):
    assert client.get('/api/health').status_code == 200


def test_missing_requires_fields(client):
    response = client.post('/api/missing', data={})
    assert response.status_code == 400
    assert 'required' in response.json['error'].lower()


def test_missing_returns_generic_success(client):
    data = {
        'fullName': 'Example Person', 'age': '30', 'gender': 'Unknown', 'lastSeenLocation': 'Indore',
        'dateTime': '2026-09-23T10:00', 'clothesDescription': 'Blue shirt', 'email': 'reporter@example.com',
        'contactNumber': '7697725054', 'photo': (image_bytes(), 'face.png'),
    }
    response = client.post('/api/missing', data=data, content_type='multipart/form-data')
    assert response.status_code == 201
    assert response.json == {'message': 'Submitted successfully. Check your email.'}


def test_public_admin_data_requires_auth(client):
    assert client.get('/api/admin/history').status_code == 401


def test_facenet512_embeddings_are_normalized_and_dimension_checked():
    service = FaceService('Facenet512')
    assert service.normalize_embedding([0.0] * 512, 'test').shape == (512,)
    assert service.cosine_distance(np.ones(512), np.ones(512)) == pytest.approx(0.0)
    with pytest.raises(FaceAnalysisError, match='Expected Facenet512 embedding'):
        service.cosine_distance(np.ones(512), np.ones(3))


def test_matching_skips_malformed_stored_embeddings():
    service = FaceService('Facenet512')
    matcher = MatchingService(None, service, None, 0.35)
    assert matcher._possible_matches(np.ones(512), [{'_id': 'legacy-bad', 'faceEmbedding': [1, 2, 3]}]) == []
