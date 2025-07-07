# face_utils.py

import os
import cv2
import numpy as np
import requests
import zipfile
from tqdm import tqdm
from insightface.app import FaceAnalysis

# ─── CONFIG ─────────────────────────────────────────────────────────────────
MODEL_NAME = 'antelopev2'
MODEL_ROOT = os.path.expanduser('~/.insightface')
MODELS_DIR = os.path.join(MODEL_ROOT, 'models')
MODEL_DIR  = os.path.join(MODELS_DIR, MODEL_NAME)

MODEL_ZIP  = os.path.join(MODELS_DIR, f'{MODEL_NAME}.zip')
MODEL_URL  = (
    'https://github.com/deepinsight/insightface/'
    'releases/download/v0.7/antelopev2.zip'
)
# ────────────────────────────────────────────────────────────────────────────

def download_and_extract_model():
    """Download & unzip the antelopev2 model pack with a progress bar."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"[+] Model '{MODEL_NAME}' not found locally. Downloading now...")

    # Stream download so we can track progress
    resp = requests.get(MODEL_URL, stream=True)
    resp.raise_for_status()
    total_size = int(resp.headers.get('content-length', 0))
    chunk_size = 8192

    with open(MODEL_ZIP, 'wb') as f, \
         tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"[+] Extracting to {MODELS_DIR} …")
    with zipfile.ZipFile(MODEL_ZIP, 'r') as z:
        z.extractall(MODELS_DIR)
    os.remove(MODEL_ZIP)
    print(f"[+] Download complete. Model is now at:\n    {MODEL_DIR}")

# ─── Ensure model directory exists ──────────────────────────────────────────
if not os.path.isdir(MODEL_DIR):
    download_and_extract_model()
# ────────────────────────────────────────────────────────────────────────────

# initialize FaceAnalysis
app = FaceAnalysis(
    name=MODEL_NAME,
    root=MODEL_ROOT,
    allowed_modules=['detection', 'recognition'],
    providers=['CPUExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))


def get_embedding(image_path):
    """
    - Reads the image.
    - Detects the face with highest score.
    - Returns a 512-dim L2-normalized embedding.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image at {image_path}")
    faces = app.get(img)
    if not faces:
        raise ValueError(f"No face detected in {image_path}")
    face = max(faces, key=lambda x: x.det_score)
    emb = face.embedding
    return emb / np.linalg.norm(emb)


def cosine_similarity(e1, e2):
    """Since embeddings are normalized, dot product == cosine similarity."""
    return float(np.dot(e1, e2))
