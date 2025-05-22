# 🌍 Multilingual Machine Translation Pipeline (English ↔ French)

This project implements and compares a progression of machine translation techniques from scratch and with pretrained models—ranging from **Hidden Markov Models (HMM)** and **LSTM-based encoder-decoder architectures** to **Transformer-based MarianMT fine-tuning**—on the [OPUS Books Dataset](https://huggingface.co/datasets/opus_books). It includes comprehensive data preprocessing, EDA, tokenization (SentencePiece), and evaluation via **BLEU scores**.

---

## 🔍 Project Highlights

| Model          | BLEU Score | Key Insight                                      |
|----------------|------------|--------------------------------------------------|
| HMM (custom)   | 0.0005     | Serves as a probabilistic baseline               |
| LSTM           | 0.62       | Captures temporal structure, handles sequences   |
| MarianMT (HF)  | 33.1       | Pretrained transformer significantly outperforms|

---

## 🧠 Features & Workflow

### 1. 📊 Exploratory Data Analysis (EDA)
- Sentence length distributions (box plots, histograms)
- Word frequency visualizations
- N-gram extraction (bigrams)
- POS tagging and sentence ratio analysis

### 2. 🧼 Preprocessing Pipeline
- HTML/markup removal, contraction expansion
- Tokenization using **spaCy** and **SentencePiece**
- Padding & encoding with custom vocabularies

### 3. 🧮 Models Implemented

#### 🔹 Hidden Markov Model (HMM)
- Built from scratch using `hmmlearn`
- Transition matrix analysis
- BLEU-based evaluation

#### 🔹 LSTM Encoder–Decoder
- Custom implementation using Keras
- Word-level tokenization with padding
- Training with teacher forcing and validation loss monitoring

#### 🔹 MarianMT Transformer
- Fine-tuned `Helsinki-NLP/opus-mt-en-fr` model using Hugging Face Transformers
- BLEU scoring using `nltk.translate`
- Vectorized attention and forced BOS token control

---

## 🧪 Evaluation

- **BLEU Score Calculation** (HMM, LSTM, Transformer)
- Examples of predictions with actual vs. generated French translations
- Outlier analysis and error propagation review

---

## 📦 Installation

```bash
pip install datasets matplotlib seaborn nltk wordcloud scikit-learn spacy contractions sentencepiece hmmlearn transformers
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
