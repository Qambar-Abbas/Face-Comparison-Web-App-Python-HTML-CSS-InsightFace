# Face Comparison Web App  
(Python, HTML, CSS, InsightFace, Antelopev2)

A minimal yet professional face comparison web app using **InsightFace** and Python's built-in HTTP server — no Flask or external web framework required.

This app allows users to upload two face images, computes cosine similarity between them, and displays a Likelihood label with visual feedback in color.

---

## Watch Demo Tutorial  
<a href="https://youtu.be/RqU0jaeowSI" target="_blank">
  <img src="https://github.com/Qambar-Abbas/Face-Comparison-Web-App-Python-HTML-CSS-InsightFace/blob/5f9dae9a9c85666a4d4a9e82d9cb4417059ade11/assets/screenshot.jpg?raw=true" alt="Watch the video" />
</a>


---

## Project Structure

```
face-compare-app/
│
├── app.py                # HTTP server
├── face_utils.py         # Embedding + cosine similarity
├── requirements.txt      # List of required packages
├── .gitignore            # Ignore unwanted files
├── README.md             # This file
│
└── static/
    ├── index.html        # Web interface
    └── style.css         # CSS styling

```

---

## Requirements

- Python 3.8+
- pip

Dependencies:
- insightface
- onnxruntime  *(or onnxruntime-gpu for GPU users)*
- opencv-python
- tqdm
- numpy
- requests

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Qambar-Abbas/Face-Comparison-Web-App-Python-HTML-CSS-InsightFace.git
cd Face-Comparison-Web-App-Python-HTML-CSS-InsightFace
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

> If using GPU:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### 4. Run the server

```bash
python app.py
```

### 5. Open in browser

Go to:  
http://localhost:8080

---

## How It Works

- Upload two images via the form
- InsightFace extracts face embeddings (512D vectors)
- Cosine similarity is computed between them
- The system classifies similarity as:

```
> 0.55   → Extremely possible
> 0.50   → Highly likely
> 0.45   → Likely
≤ 0.45   → Unlikely
```

- Likelihood is shown in color (green → red)

---

## Features

- Responsive card-based UI
- No JavaScript or Flask required
- Face likelihood with color-coded labels
- Real-time image preview after upload
- Semantic HTML5 + CSS3 only

---

## Notes

- Images are temporarily saved and cleaned after processing
- Uses antelopev2 model from InsightFace
- Model is downloaded to `~/.insightface` on first run

---

## License

Licensed under the MIT License.

---

## Author

Created by Qambar Abbas  
Built with Python, InsightFace, HTML/CSS
