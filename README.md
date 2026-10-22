<div align="center">

# 🧠 Next Word Prediction with LSTM & GRU

**A deep learning NLP project that predicts the next word in a sentence using recurrent neural networks, served through a modern Streamlit web app.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

[Overview](#-overview) •
[Features](#-features) •
[Tech Stack](#-tech-stack) •
[Project Structure](#-project-structure) •
[Getting Started](#-getting-started) •
[How It Works](#-how-it-works) •
[Roadmap](#-roadmap) •
[Contact](#-contact)

</div>

---

## 📖 Overview

Next word prediction is a core task in Natural Language Processing, powering autocomplete, smart keyboards, and writing assistants. This project builds and compares two recurrent architectures, **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit), to predict the most likely next word for a given text sequence.

The trained model is wrapped in a clean, responsive **Streamlit** interface so anyone can try it in the browser with no code required.

## ✨ Features

- 🔤 **Next word prediction** from any input sequence
- 🧬 **Two architectures:** LSTM and GRU models for comparison
- ⏹️ **Early stopping** during training to prevent overfitting
- 🎨 **Modern dark UI** built with Streamlit and custom CSS
- 💾 **Saved model and tokenizer** for fast inference without retraining
- 📊 **Model info panel** showing input length and vocabulary size

## 🛠️ Tech Stack

| Category | Tools |
| --- | --- |
| Language | Python |
| Deep Learning | TensorFlow, Keras (LSTM, GRU, Embedding, Dense) |
| NLP Preprocessing | Keras Tokenizer, `pad_sequences` |
| Numerical Computing | NumPy |
| Web App | Streamlit |
| Serialization | Pickle, HDF5 (`.h5`) |

## 📂 Project Structure

```text
Next_word_prediction_with_GRU_and_LSTM/
├── app.py                  # Streamlit web application
├── next_word_lstm.h5       # Trained LSTM model
├── tokenizer.pickle        # Fitted tokenizer
├── requirements.txt        # Python dependencies
├── notebooks/              # Training & experimentation notebooks (LSTM / GRU)
└── README.md
```

> 📝 Adjust the tree above so it matches the actual files in your repository.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (or conda)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Madhavkumaryadav/Next_word_prediction_with_GRU_and_LSTM.git
cd Next_word_prediction_with_GRU_and_LSTM
```

**2. Create and activate a virtual environment (recommended)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, create one with:

```text
streamlit
tensorflow
numpy
```

**4. Run the app**

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## 🎯 Usage

1. Type a few words into the input box, for example `To be or not to`.
2. Click **✨ Predict the next word**.
3. The predicted word appears in the result card.

| Input | Predicted output |
| --- | --- |
| `To be or not to` | *model prediction* |

## 🧠 How It Works

```text
Raw Text → Tokenization → N-gram Sequences → Padding → Embedding → LSTM / GRU → Dense (Softmax) → Next Word
```

1. **Tokenization:** text is converted into integer sequences using the Keras `Tokenizer`.
2. **Sequence generation:** n-gram sequences are created so the model learns word context.
3. **Padding:** sequences are padded to equal length with `pad_sequences`.
4. **Model training:** an Embedding layer feeds an LSTM or GRU layer, followed by a Dense softmax layer over the vocabulary.
5. **Early stopping:** training halts when validation loss stops improving.
6. **Inference:** the input text is tokenized, padded, passed through the model, and the predicted index is mapped back to a word.

### LSTM vs GRU

| | LSTM | GRU |
| --- | --- | --- |
| Gates | 3 (input, forget, output) | 2 (update, reset) |
| Parameters | More | Fewer |
| Training speed | Slower | Faster |
| Long-range memory | Very strong | Strong |

## 🗺️ Roadmap

- [ ] Show top-3 / top-5 predictions with confidence scores
- [ ] Add the GRU model as a selectable option in the app
- [ ] Train on a larger, more diverse corpus
- [ ] Add a Transformer-based baseline for comparison
- [ ] Deploy on Streamlit Community Cloud

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 📬 Contact

**Madhav Yadav**

[![Email](https://img.shields.io/badge/Email-ymadhav081%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:ymadhav081@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Madhavkumaryadav-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Madhavkumaryadav)

---

<div align="center">

⭐ If you found this project useful, please consider giving it a star!

© 2026 All rights reserved by Madhav Yadav

</div>
