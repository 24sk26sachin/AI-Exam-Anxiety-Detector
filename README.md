# AI-Based Exam Anxiety Detector

## Overview

The AI-Based Exam Anxiety Detector is an intelligent mental-wellness support system designed to identify and categorize exam-related anxiety from student-generated text inputs. The system leverages Natural Language Processing (NLP) and a BERT-based deep learning model to analyse linguistic patterns associated with stress, fear, nervousness, and emotional pressure.

## Features

- **Real-time Anxiety Classification**: Categorizes anxiety into Low, Moderate, and High levels.
- **FastAPI Backend**: Handles text input and provides model inference.
- **Streamlit Frontend**: Intuitive user interface with visual indicators and anxiety-management tips.
- **Supportive Tool**: Designed as a non-diagnostic tool to assist students and educators.

## Architecture

- **Backend**: FastAPI, Hugging Face Transformers (BERT).
- **Frontend**: Streamlit.
- **Model**: Fine-tuned BERT for classification.

## Getting Started

### Prerequisites

- Python 3.9+
- Virtual Environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone <GitHub-Link>
   cd "AI Based Exam Anxiety Detector"
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Backend (FastAPI)

```bash
uvicorn backend.main:app --reload
```

#### Frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

## 🔗 Project Links

> [!IMPORTANT]
> Please update these links for your mentor's review.

- **Live Demo**: [🚀 Click here to see the Demo](https://ai-exam-anxiety-detector-byteforce.streamlit.app/)
- **GitHub Repository**: [💻 Visit the Source Code](https://github.com/24sk26sachin/AI-Exam-Anxiety-Detector/)

## License

MIT License
