# 🧬 Bio-Identity Analytics System

An advanced biometric classification system that uses Deep Learning (VGG-Face) and Computer Vision to detect Nationality, Age, Emotion, and Fashion Attributes based on dynamic business logic.

> Note: This project is intended for educational and research purposes. See “Ethics & Fair Use” for important considerations.

---

## 🚀 Key Features
- **Smart Logic:** Custom visibility rules for Indian, USA, African, and Other nationalities.
- **Computer Vision:** K-Means clustering & KD-Tree spatial lookup for dress color analysis.
- **Deep Learning:** Face analysis with [DeepFace](https://github.com/serengil/deepface) (VGG-Face backbone).
- **Interactive UI:** Streamlit interface with live previews and charts.
- **Visual Analytics:** Plotly-based charts for distributions and attribute insights.
- **Modular Pipeline:** Separate components for face detection, attribute inference, color analytics, and business rules.

---

## 🧩 What It Detects
- Nationality (rule-based visibility and classification display logic)
- Age group estimation
- Emotion (e.g., happy, neutral, sad, etc.)
- Fashion attributes:
  - Dominant dress/top color via K-Means clustering
  - Nearest named color via KD-Tree lookup (e.g., “Navy”, “Maroon”)

---

## 🏗️ Architecture Overview
1. Image upload or camera capture via Streamlit.
2. Face detection and embedding via DeepFace (VGG-Face).
3. Attribute inference:
   - Age and emotion from DeepFace analyzers
   - Dress color via K-Means clustering on detected clothing region (or ROI)
   - Color naming via KD-Tree over a predefined palette (e.g., CSS/X11 colors)
4. Business logic layer enforces attribute visibility based on nationality rules.
5. Visualization with Plotly + Streamlit components.

---

## 📦 How to Run

### 1) Prerequisites
- Python 3.9+ recommended
- pip installed
- On Windows, install build tools (if compiling dependencies)
- On Linux/macOS, ensure OpenCV dependencies are available

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

If you encounter issues with `opencv-python` or `deepface`, try:
```bash
pip install opencv-python-headless
pip install deepface
```

### 3) Run the app
```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal (e.g., http://localhost:8501). Open it in your browser.

---

## ⚙️ Configuration & Business Rules

You can adjust nationality-specific visibility rules inside the business logic layer (typically a helper or config section in your app code). Example approach:

- Indian: show age, emotion, and fashion attributes
- USA: show emotion and fashion attributes only
- African: show age and emotion only
- Other: show basic attributes (emotion) and restrict sensitive outputs

If you externalize rules to a config (e.g., `config/rules.yaml`), you can define:
```yaml
visibility_rules:
  indian:
    show_age: true
    show_emotion: true
    show_fashion: true
  usa:
    show_age: false
    show_emotion: true
    show_fashion: true
  african:
    show_age: true
    show_emotion: true
    show_fashion: false
  other:
    show_age: false
    show_emotion: true
    show_fashion: false
```

Then load these into your Streamlit session and apply in the UI when rendering results.

---

## 👗 Fashion Color Analytics

- Extract dominant color from a clothing ROI using K-Means (k=3–5 recommended).
- Convert centroid RGB to Lab for perceptual uniformity (optional).
- Use a KD-Tree built over a named color palette (CSS/X11) to find the closest human-readable color.

Outputs:
- Dominant color swatch
- Named color (e.g., “Royal Blue”)
- Confidence or distance metric to indicate match quality

---

## 📊 Visualizations
- Age distribution by detected class (if batch or multi-input)
- Emotion pie chart or bar chart
- Color swatches with labels
- Per-nationality visibility matrix (what’s shown vs hidden)

---

## 🖥️ Usage Flow
1. Upload an image or capture from the camera component.
2. The system detects faces and runs attribute inference.
3. Business logic determines which attributes to show.
4. Results are displayed with charts and color swatches.
5. Optionally, export results (CSV/JSON) for audits or batch processing.

---

## 📁 Suggested Project Structure

If you plan to modularize the project, a typical layout might be:
```
bio-identity-system/
├─ app.py
├─ requirements.txt
├─ src/
│  ├─ pipeline/
│  │  ├─ face.py
│  │  ├─ attributes.py
│  │  ├─ color.py
│  │  └─ rules.py
│  ├─ ui/
│  │  ├─ components.py
│  │  └─ charts.py
│  └─ utils/
│     ├─ kmeans.py
│     ├─ kd_tree.py
│     └─ colors.py
├─ config/
│  └─ rules.yaml
├─ assets/
│  ├─ palette.json
│  └─ samples/
└─ tests/
   ├─ test_face.py
   ├─ test_attributes.py
   └─ test_color.py
```

Note: Adjust to match your actual repository files.

---

## 🧪 Testing
- Unit tests for each pipeline component (face, attributes, color).
- Add sample images under `assets/samples/`.
- Use `pytest` to run tests:
```bash
pytest -q
```

---

## 🐳 Optional: Docker
Build and run for a consistent environment:
```bash
docker build -t bio-identity-system .
docker run -p 8501:8501 bio-identity-system
```

---

## 🔧 Troubleshooting
- If DeepFace fails to load models, ensure internet access the first time (models cache locally).
- For OpenCV GUI issues on servers, use `opencv-python-headless`.
- If Streamlit doesn’t open, verify the port isn’t in use or set a custom port:
```bash
streamlit run app.py --server.port=8502
```

---

## 🔒 Ethics & Fair Use
This system performs sensitive biometric analysis. Use responsibly:
- Avoid discrimination, bias, or unjust profiling.
- Ensure consent for data collection and processing.
- Comply with local data protection regulations (e.g., GDPR).
- Consider cultural, ethical, and fairness implications of nationality and emotion inference.

---

## 📚 References
- DeepFace: [https://github.com/serengil/deepface](https://github.com/serengil/deepface)
- VGG-Face: Parkhi et al., “Deep Face Recognition”
- Color naming resources: CSS/X11 color lists and Lab color space references

---

## 📄 License
Specify a license (e.g., MIT) in `LICENSE`. If absent, default is “All rights reserved.”

---

## 🙌 Acknowledgments
Thanks to the open-source community behind DeepFace, OpenCV, Streamlit, Plotly, and Scikit-Learn.

---

## 🛣️ Roadmap
- Improve nationality classification calibration and transparency tools
- Add multi-face support with per-face segmentation
- Expand fashion attributes (texture, pattern detection)
- Add batch processing mode and exports
- Integrate explainability dashboards (embeddings and attribution)

---
