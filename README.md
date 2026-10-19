# Customer Review Intelligence System

An AI-powered machine learning and natural language processing application that analyzes unstructured customer reviews, classifies sentiment with calibrated confidence scoring, semantically groups similar complaints together into distinct issue clusters regardless of phrasing, extracts representative customer quotes (medoids), and delivers actionable analytics via an interactive visual dashboard.

Built with Python 3.11, Scikit-Learn, FastEmbed (Microsoft ONNX Runtime), VADER, and Streamlit. Runs 100% locally on standard CPU with zero paid cloud API fees.

---

## 1. Project Objective

Customer feedback contains critical signals about product defects, logistics bottlenecks, software bugs, and customer service failures. However, traditional analytics tools rely on fixed keyword matching, which fails when customers describe the same underlying problem using diverse vocabulary:
* *"Battery drains too quickly"*
* *"The charge doesn't last through my morning"*
* *"My phone dies within hours"*

The **Customer Review Intelligence System** transforms unstructured reviews into structured, traceable intelligence using dense semantic embeddings and hierarchical clustering, presenting results in an intuitive visual dashboard with interactive charts, pipeline flowcharts, category icons, and human-in-the-loop topic refinement.

---

## 2. System Architecture & Pipeline Workflow

```
[ User Input: CSV Upload, Curated Sample Dataset, or Manual Review Entry ]
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Data Ingestion & Sanitization Layer                                      │
│    - Multi-encoding CSV parser (UTF-8, Latin-1, CP1252)                     │
│    - Automatic review column detection with manual override                 │
│    - Rejection of empty rows, nulls, and trivial fragments (< 10 chars)     │
│    - Standardized 0-indexed review traceability IDs                         │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. NLP & Inference Pipeline                                                 │
│    ├── A. Sentiment Classification & Calibration                            │
│    │      - 3-class classification (Positive, Neutral, Negative)            │
│    │      - Continuous compound valence score (-1.0 to +1.0) & confidence   │
│    │      - Domain lexicon tuning for product defects (draining, crashing)  │
│    │      - Overall product verdict override logic                          │
│    │                                                                        │
│    └── B. Semantic Clustering Engine                                        │
│           - Dense vector representation: FastEmbed ONNX (384 dimensions)     │
│           - Semantic grouping: Cosine Agglomerative Hierarchical Clustering  │
│           - Automated issue naming: Class-based TF-IDF (c-TF-IDF)           │
│           - Mathematical centroid & central medoid quote extraction         │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. State & Human-in-the-Loop Refinement Layer                               │
│    - Session state tracking original vs customized business labels          │
│    - User ability to rename, edit, or reassign issue topics                 │
│    - Granularity sensitivity slider (distance threshold)                    │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Interactive Visual Dashboard & Export Layer                              │
│    - Executive metric strip with status badges and icons                    │
│    - Interactive Altair charts: Issue ranking, sentiment donut, composition │
│    - Single Review Deep Inspection with stacked visual polarity meters      │
│    - Traceable review explorer with keyword, sentiment, and topic filters   │
│    - One-click CSV export with all enriched model outputs                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Features & Visual Components

### 📊 Interactive Visual Analytics
* **AI Pipeline Architecture Flowchart:** Expandable visual diagram showing the 6-step data science process from ingestion to actionable analytics.
* **Interactive Altair Charts:**
  * **Issue Frequency Ranking:** Horizontal bar chart sorting problem areas by review count.
  * **Sentiment Distribution Donut:** Visual breakdown of Negative, Neutral, and Positive proportions.
  * **Per-Issue Sentiment Breakdown:** Stacked composition bar chart highlighting defect severity per cluster.
* **Contextual Category Icons:** Automatically tags topics with intuitive icons (`🔋 Battery`, `📦 Shipping`, `💻 App & Software`, `🎧 Audio & Bluetooth`, `⚙️ Hardware & Build`, `💬 Customer Support`, `🌟 Praise`).
* **Visual Polarity Meters:** Stacked color-coded bars indicating negative, neutral, and positive score proportions.

### ✍️ Flexible Ingestion Modes
1. **Curated Sample Dataset:** 61 realistic customer reviews across hardware, software, battery, shipping, and support.
2. **Custom CSV Upload:** Upload any real-world review CSV; automatic column detector selects review text with manual fallback.
3. **Manual Review Entry (Single or Multi-line):**
   * *Single review:* Deep inspection card, classification rationale with model notes, nearest knowledge base historical match, and visual polarity breakdown.
   * *Multi-line reviews:* Paste multiple reviews (one per line) for on-the-fly semantic clustering and charting.

### 🛠️ Human-in-the-Loop Topic Management
* Inspect algorithmic topic names and rename them using business-friendly terms.
* Changes instantly synchronize across all dashboard charts, tables, and the export CSV.

---

## 4. Machine Learning & NLP Methodology

### Semantic Vector Representation
* **Model:** `BAAI/bge-small-en-v1.5` via **FastEmbed** and **Microsoft ONNX Runtime**.
* **Vector Dimensionality:** 384 dense floating-point dimensions.
* **Why this approach?** Dense sentence transformers capture semantic context geometrically. Two phrases with zero overlapping words (*"charge doesn't hold"* and *"battery runs out fast"*) achieve **>0.70 cosine similarity**, enabling true semantic clustering that keyword search cannot replicate.
* **Hardware Efficiency:** Executes 100% locally on standard laptop CPUs in milliseconds, requiring zero external paid API keys or dedicated GPUs.

### Issue Grouping & Representative Quotes
* **Clustering:** Agglomerative Hierarchical Clustering with cosine distance thresholding (and HDBSCAN density grouping).
* **Automated Topic Naming:** Employs **Class-based TF-IDF (c-TF-IDF)** to identify n-grams that are uniquely frequent within a specific issue group relative to the rest of the corpus.
* **Representative Medoid Quote:** For each issue, the system mathematically calculates the vector centroid and selects the actual customer review closest to that center (the medoid).

### Sentiment Analysis
* **Engine:** Calibrated **VADER** (Valence Aware Dictionary and sEntiment Reasoner).
* **Calibration:** Augmented with domain-specific customer review vocabulary (*drains*, *crashes*, *overheating*, *unpairs*, *delayed*, *flawlessly*).
* **Verdict Override Logic:** Correctly flags overall sentiment when mixed phrasing is present (e.g., *"Bad product overall but good battery health"* -> classified as **Negative** with 84% confidence).

---

## 5. Documented Model Limitations

To maintain scientific integrity and transparent engineering:
1. **Subtle Sarcasm & Irony:** Lexicon-based sentiment models can struggle with complex sarcasm (e.g., *"Oh wonderful, another broken hinge"*).
2. **Evenly Balanced Feedback:** Reviews containing equal volumes of praise and criticism may balance near zero compound polarity and be categorized as Neutral.
3. **Multi-Issue Reviews:** If a single customer lists four distinct problems in one sentence, the embedding reflects the composite semantic centroid.
4. **Human-in-the-Loop Solution:** The application provides a dedicated **Label Management** tab, allowing business stakeholders to inspect and adjust algorithmic labels before exporting.

---

## 6. How to Run Locally

### Prerequisites
* Windows 10/11 or macOS/Linux
* Python 3.10 or 3.11

### Step 1: Open Terminal in the Project Directory
```powershell
cd C:\Users\abdsa\.gemini\antigravity\scratch\customer-review-intelligence
```

### Step 2: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(On macOS/Linux: `source .venv/bin/activate`)*

### Step 3: Run the Streamlit Application
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 7. Running Automated Tests & Evaluation

### Run the Full Unit Test Suite (22 Automated Tests)
```powershell
python -m pytest -v
```
Verifies CSV ingestion, missing/null handling, short-row rejection, column detection, 3-class sentiment polarity, negation handling, contrastive conjunctions, ONNX vector embeddings, semantic clustering, and c-TF-IDF topic labeling.

### Run End-to-End Pipeline Verification
```powershell
python verify_sample_pipeline.py
```
Executes the full pipeline on 61 real customer reviews and prints discovered issues and representative quotes in the terminal.

### Run Ground-Truth Benchmark Evaluation
```powershell
python evaluate.py
```
Evaluates accuracy, precision, recall, Adjusted Rand Index (ARI: 0.709), and Normalized Mutual Information (NMI: 0.904) against the manually verified ground-truth test set.

---

## 8. Free Deployment Guide

### Option A: Streamlit Community Cloud (Recommended)
1. Push this project folder to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Customer Review Intelligence System"
   git remote add origin https://github.com/YOUR_USERNAME/customer-review-intelligence.git
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **New app**, select your repository and branch (`main`), set the main file path to `app.py`, and click **Deploy**.
4. The application will build and launch at a public URL (e.g., `https://your-app.streamlit.app`).

### Option B: Hugging Face Spaces (Free CPU Tier)
1. Create a free account at [huggingface.co](https://huggingface.co).
2. Go to **Spaces** -> **Create new Space**.
3. Select the **Streamlit** SDK and the free **CPU basic (2 vCPU, 16 GB RAM)** hardware tier.
4. Push your repository code to the Space's Git remote. Hugging Face will automatically detect `requirements.txt` and launch the application.

---

## 9. Project File Structure

```
customer-review-intelligence/
├── data/
│   ├── sample_reviews.csv       # 61 realistic reviews across 6 customer issue categories
│   └── test_ground_truth.csv    # 17 manually verified ground-truth benchmark reviews
├── src/
│   ├── __init__.py              # Package init with Windows ONNX runtime DLL pre-load
│   ├── config.py                # Paths, thresholds, candidate column names, color constants
│   ├── data_loader.py           # Multi-encoding CSV loader, validation, and single review wrapper
│   ├── sentiment.py             # Calibrated VADER 3-class sentiment analyzer with confidence scores
│   ├── embeddings.py            # FastEmbed ONNX dense vector embedding generator (384-d)
│   ├── clustering.py            # Agglomerative & HDBSCAN semantic clustering and medoid calculation
│   ├── issue_extractor.py       # c-TF-IDF keyword extraction and automated issue naming
│   └── ui_components.py         # Responsive CSS, Altair visual charts, pipeline diagram, and icons
├── tests/
│   ├── conftest.py              # Pytest configuration with ONNX DLL pre-binding
│   ├── test_data_loader.py      # Unit tests for data ingestion & validation (8 tests)
│   ├── test_sentiment.py        # Unit tests for sentiment classification & negation (8 tests)
│   └── test_clustering.py       # Unit tests for embeddings & clustering (6 tests)
├── app.py                       # Main Streamlit application entrypoint with full interactive dashboard
├── evaluate.py                  # Ground-truth evaluation script (Accuracy, ARI, NMI, Silhouette)
├── verify_sample_pipeline.py    # Terminal verification script for full pipeline
├── pytest.ini                   # Pytest configuration
├── requirements.txt             # Pinned project dependencies
└── README.md                    # System documentation, architecture, and deployment guide
```
