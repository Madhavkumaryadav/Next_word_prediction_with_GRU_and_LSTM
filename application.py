import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------------------------
# Page configuration (UI only)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Next Word Prediction | LSTM",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

## load the LSTM Model
# model = load_model('high_accuracy_model.h5')
model = load_model('next_word_lstm.h5')
# load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


## Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  ## Ensure the sequence length matches
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_word_index = np.argmax(token_list, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


# ---------------------------------------------------------------------------
# Styling (UI only)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    .stApp {
        background:
            radial-gradient(1200px 500px at 15% -10%, rgba(99,102,241,0.18), transparent 60%),
            radial-gradient(900px 500px at 100% 0%, rgba(14,165,233,0.14), transparent 55%),
            #0b1020;
        color: #e5e7eb;
    }
    .block-container { padding-top: 2.5rem; max-width: 760px; }

    /* Hero */
    .hero { text-align: center; margin-bottom: 1.8rem; }
    .badge {
        display: inline-block; padding: 6px 14px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: .06em;
        color: #a5b4fc; background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.35); margin-bottom: 14px;
    }
    .hero h1 {
        font-size: 2.6rem; font-weight: 800; line-height: 1.15; margin: 0 0 10px 0;
        background: linear-gradient(90deg, #ffffff 0%, #a5b4fc 55%, #38bdf8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { color: #94a3b8; font-size: 1.02rem; margin: 0; }

    /* Card */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px; padding: 26px 26px 8px 26px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.35);
        backdrop-filter: blur(10px);
    }
    .label {
        font-size: 0.78rem; font-weight: 600; letter-spacing: .08em;
        text-transform: uppercase; color: #94a3b8; margin-bottom: 6px;
    }

    /* Input */
    .stTextInput input {
        background: rgba(15,23,42,0.85) !important; color: #f1f5f9 !important;
        border: 1px solid rgba(148,163,184,0.25) !important; border-radius: 12px !important;
        padding: 14px 16px !important; font-size: 1.05rem !important;
    }
    .stTextInput input:focus {
        border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.25) !important;
    }
    .stTextInput label { display: none; }

    /* Button */
    .stButton > button {
        width: 100%; border: none; border-radius: 12px; padding: 0.85rem 1rem;
        font-weight: 700; font-size: 1rem; color: #fff;
        background: linear-gradient(90deg, #6366f1, #0ea5e9);
        transition: transform .15s ease, box-shadow .15s ease;
        box-shadow: 0 10px 25px rgba(99,102,241,0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px); color: #fff;
        box-shadow: 0 14px 30px rgba(99,102,241,0.5);
    }
    .stButton > button:active { transform: translateY(0); }

    /* Result */
    .result {
        margin-top: 22px; padding: 22px; border-radius: 16px; text-align: center;
        background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(14,165,233,0.12));
        border: 1px solid rgba(99,102,241,0.4);
    }
    .result .sentence { color: #cbd5e1; font-size: 1.1rem; margin-bottom: 10px; }
    .result .sentence b {
        color: #fff; background: linear-gradient(90deg,#6366f1,#0ea5e9);
        padding: 2px 12px; border-radius: 8px;
    }
    .result .small { color: #94a3b8; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
    .result .word { font-size: 2.1rem; font-weight: 800; color: #fff; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0a0f1e; border-right: 1px solid rgba(255,255,255,0.07);
    }
    .side-title { font-weight: 700; color: #f1f5f9; font-size: 1.05rem; margin-bottom: 4px; }
    .side-text { color: #94a3b8; font-size: .88rem; line-height: 1.55; }
    .stat {
        display: flex; justify-content: space-between; padding: 10px 14px; margin-top: 8px;
        border-radius: 10px; background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07); font-size: .85rem; color: #cbd5e1;
    }
    .stat span:last-child { color: #a5b4fc; font-weight: 600; }

    /* Footer */
    .footer {
        text-align: center; margin-top: 3rem; padding-top: 1.2rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        color: #64748b; font-size: .82rem;
    }
    .footer a { color: #a5b4fc; text-decoration: none; }
    .footer a:hover { text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-title">🧠 About this app</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-text">An LSTM-based language model that suggests the most '
        'likely next word for the text you type. Trained with early stopping to '
        'avoid overfitting.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="side-title">⚙️ Model info</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="stat"><span>Architecture</span><span>LSTM</span></div>
        <div class="stat"><span>Input length</span><span>{model.input_shape[1]} tokens</span></div>
        <div class="stat"><span>Vocabulary</span><span>{len(tokenizer.word_index):,} words</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="side-title">💡 Tips</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-text">• Type a few words of a sentence<br>'
        '• Longer context usually gives better results<br>'
        '• Try famous lines, e.g. <i>"To be or not to"</i></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="badge">DEEP LEARNING · NLP</div>
        <h1>Next Word Prediction</h1>
        <p>Powered by an LSTM neural network with early stopping</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="label">Enter your sequence of words</div>', unsafe_allow_html=True)
input_text = st.text_input(
    "Enter the sequence of word ",
    "To be or not to",
    placeholder="Start typing a sentence...",
)
predict_clicked = st.button("✨  Predict the next word")
st.markdown('</div>', unsafe_allow_html=True)

if predict_clicked:
    if not input_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing sequence..."):
            max_sequence_len = model.input_shape[1] + 1
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.markdown(
            f"""
            <div class="result">
                <div class="sentence">{input_text} <b>{next_word}</b></div>
                <div class="small">Predicted next word</div>
                <div class="word">{next_word}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="footer">
        © 2026 All rights reserved by <b>Madhav Yadav</b> ·
        <a href="mailto:ymadhav081@gmail.com">ymadhav081@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True,
)