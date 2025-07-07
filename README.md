# 🔍 Face Comparison Web App  (Python, HTML, CSS, InsightFace, Antelopev2)

A minimal yet professional face comparison web app using **InsightFace** and Python's built-in HTTP server — no Flask or external web framework required.

This app allows users to upload two face images, computes cosine similarity between them, and displays a Likelihood label with visual feedback in color.

---

## 📸 Interface Preview

🎥 **Watch Demo Tutorial**  
[![Video Tutorial](https://github.com/Qambar-Abbas/Face-Comparison-Web-App-Python-HTML-CSS-InsightFace/blob/f20574f95e5886a3da45a779ab5655e297ae928f/images/videos/Screenshot%202025-07-08%20at%201.30.11%E2%80%AFAM.png)](https://github.com/Qambar-Abbas/Face-Comparison-Web-App-Python-HTML-CSS-InsightFace/blob/f20574f95e5886a3da45a779ab5655e297ae928f/images/videos/Screen%20Recording%202025-07-08%20at%2012.58.41%20AM%20(1)%20(1).mp4)


---

## 📁 Project Structure

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
    ├── style.css         # CSS styling
    └── preview.png       # Interface screenshot
```

---

## ✅ Requirements

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

## ⚙️ Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/your-username/face-compare-app.git
cd face-compare-app
```

### 2. Create and activate a virtual environment

```
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install all dependencies

```
pip install -r requirements.txt
```

> If using GPU:
```
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### 4. Run the server

```
python app.py
```

### 5. Open in browser

Go to:  
http://localhost:8080

---

## 💡 How It Works

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

## 🎨 Features

- Responsive card-based UI
- No JavaScript or Flask required
- Face likelihood with color-coded labels
- Real-time image preview after upload
- Semantic HTML5 + CSS3 only

---

## 📌 Notes

- Images are temporarily saved and cleaned after processing
- Uses antelopev2 model from InsightFace
- Model is downloaded to `~/.insightface` on first run

---

## 📄 License

Licensed under the MIT License.

---

## 🙋‍♂️ Author

Created by Qambar Abbas  
Built with ❤️ Python, 💡 InsightFace, and ⚡ HTML/CSS
