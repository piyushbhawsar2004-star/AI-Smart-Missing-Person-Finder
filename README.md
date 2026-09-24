# AI Smart Missing Person Finder
live - https://ai-smart-missing-person-finder.onrender.com/
AI-assisted missing and found person reporting for communities, with private notifications and mandatory human verification.

## Features

- Missing-person gallery upload and found-person mobile camera capture.
- DeepFace face detection and Facenet512 embeddings.
- Configurable cosine-distance threshold for possible matches.
- Matching in both directions: found -> active missing and new missing -> eligible unmatched found.
- Gmail SMTP notifications without hard-coded credentials.
- Private uploads served only to authenticated administrators.
- Admin login, possible-match review, confirmation/rejection, history, and audit logs.
- Confirmed records remain in MongoDB and move from `missing` to `found`; they are never deleted.

## Architecture

Flask serves REST APIs and the responsive vanilla HTML/CSS/JavaScript frontend. MongoDB stores reports, embeddings, users, match history, and audit events. DeepFace detects exactly one face and creates a Facenet512 embedding. Matching uses cosine distance; the default threshold is `0.35`, configurable with `FACE_MATCH_THRESHOLD`. A cosine similarity representation is recorded as `1 - distance`; it is not a probability or identity claim.

## Requirements

- Python 3.10 or 3.11 recommended on Windows for the broadest DeepFace/TensorFlow compatibility. Python 3.13 is supported with the TensorFlow 2.20 pin in this project.
- MongoDB 6+ running locally or a MongoDB Atlas URI.
- Gmail account with 2-Step Verification and a Gmail App Password.
- 8 GB RAM recommended for TensorFlow/DeepFace model loading.

## Installation

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

On Windows, TensorFlow can fail with a long-path error when the workspace is nested deeply under `AppData`. Use a short virtual-environment path instead:

```powershell
py -3.13 -m venv C:\mpf-venv
C:\mpf-venv\Scripts\python.exe -m pip install --upgrade pip
C:\mpf-venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run all later commands with `C:\mpf-venv\Scripts\python.exe` when using that environment.

Edit `.env`:

```env
MONGO_URI=mongodb://localhost:27017/missing_person_finder
EMAIL_USER=your-gmail-address@gmail.com
EMAIL_APP_PASSWORD=the-16-character-gmail-app-password
SECRET_KEY=generate-a-long-random-secret
FACE_MODEL=Facenet512
FACE_MATCH_THRESHOLD=0.35
UPLOAD_FOLDER=uploads
MAX_UPLOAD_MB=8
SESSION_COOKIE_SECURE=false
```

Never use the normal Gmail password and never commit `.env`.

Create the first admin:

```powershell
python seed_admin.py
```

Start the application:

```powershell
python app.py
```

Open <http://127.0.0.1:5000>. Admin sign-in is at `/` -> Admin. The frontend is served by Flask, so no separate frontend server is required.

## Testing

Run the automated request and validation checks:

```powershell
pytest -q
```

For a manual end-to-end test:

1. Start MongoDB and configure `.env`.
2. Configure Gmail App Password and create an admin.
3. Submit a clear single-face gallery image on Report Missing Person and confirm the registration email.
4. Open Report Found Person on a phone over a trusted local network/HTTPS and use the camera capture control.
5. Confirm the public result says only `Result has been sent to your email`.
6. Sign in as admin, review the possible match or no-match record, and confirm/reject from Possible matches.
7. Confirmed records appear under History and are no longer active matching candidates.

Camera capture is controlled by the browser and device. `capture="environment"` requests a rear camera where supported but cannot guarantee that behavior on every desktop/browser. For local phone testing, use a secure tunnel or a development HTTPS setup because camera permissions commonly require a secure context.

## API overview

- `POST /api/missing`
- `POST /api/found`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/admin/missing`
- `GET /api/admin/found`
- `GET /api/admin/matches`
- `GET /api/admin/history`
- `POST /api/admin/matches/<id>/confirm`
- `POST /api/admin/matches/<id>/reject`
- `GET /api/health`

Public submission responses intentionally never include AI results, names of other people, scores, embeddings, or private details.

## Security notes

Uploads are size-limited, extension/MIME checked, decoded with Pillow, UUID-renamed, and stored outside the static directory. Original filenames are not trusted. Embeddings are excluded from admin JSON responses. Admin passwords are stored as Werkzeug hashes, sessions are HTTP-only and same-site, and protected image routes require an active admin session. Configure `SESSION_COOKIE_SECURE=true` behind HTTPS.

Email failures do not roll back saved records; they are logged and written to `audit_logs` for admin investigation. Stack traces are logged server-side only.

## Troubleshooting

- **MongoDB unavailable:** start the MongoDB service or replace `MONGO_URI` with an Atlas URI; `/api/health` reports the degraded state.
- **DeepFace/TensorFlow install issues:** use Python 3.10/3.11 in a fresh virtual environment and install the pinned requirements, including `tf-keras`. On Windows, use the short `C:\mpf-venv` path if long-path errors occur. The first embedding downloads the Facenet512 model.
- **No face / multiple faces:** use a well-lit image containing exactly one clear face.
- **No email:** verify the Gmail App Password, `EMAIL_USER`, 2-Step Verification, and that SMTP port 465 is permitted.
- **Camera unavailable:** use a supported mobile browser, grant camera permissions, and test through HTTPS or localhost.

## Embedding migration

All new Missing and Found records must contain a numeric `Facenet512` embedding with exactly 512 values. Inspect existing records with:

```powershell
C:\mpf-venv\Scripts\python.exe migrate_embeddings.py
```

Malformed legacy records are logged and can be explicitly excluded from matching with:

```powershell
C:\mpf-venv\Scripts\python.exe migrate_embeddings.py --quarantine
```

The migration does not delete records or invent replacement embeddings. It marks malformed records with `matchingExcluded=true`; re-submit a clear image to create a valid replacement report.
