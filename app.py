import streamlit as st
import json
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Streamlit Config ---
st.set_page_config(page_title="Cue.ai English Tutor", page_icon="📘", layout="wide")

# --- Custom CSS for styling ---
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #f7f9fc;
        font-family: "Segoe UI", sans-serif;
    }

    /* Header container */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }

    .header-logo {
        height: 60px;
        border-radius: 8px;
    }

    h1 {
        color: #1b263b;
        text-align: center;
        font-size: 2.2rem !important;
    }

    /* Chat styling */
    .chat-bubble {
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        max-width: 85%;
    }

    .user {
        background-color: #d0e3ff;
        margin-left: auto;
    }

    .ai {
        background-color: #ffffff;
        border: 1px solid #dbe4ee;
    }

    /* Input styling */
    input, textarea {
        border-radius: 8px !important;
        border: 1px solid #d0d7de !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# --- Logo and title ---
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image("assets/logo.png", width=80)
    st.markdown("<h1 style='text-align:center; color:#1b263b;'>Cue.ai — English Learning Tutor</h1>", unsafe_allow_html=True)



# --- OpenAI client (new API) ---
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- Load chapters ---
with open("chapters.json", "r") as f:
    books = json.load(f)

chapter_titles = [c["title"] for c in books]

# --- Streamlit UI ---
st.title("AI English Tutor")

chapter_choice = st.selectbox("Select chapter", chapter_titles, key="chapter_select")
user_msg = st.text_input("You:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Embedding model ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# --- Retrieve chapter info ---
chapter = next(c for c in books if c["title"] == chapter_choice)

chapter_text = (
    chapter["description"] + ". "
    + " ".join(chapter["vocabulary"]) + ". "
    + " ".join(chapter["grammar"]) + ". "
    + " ".join(chapter["sample_dialogues"])
)

# --- Vectorize chapter chunks ---
@st.cache_data
def get_chunk_embeddings(chapter_text):
    sentences = chapter_text.split(". ")
    embeddings = embedding_model.encode(sentences)
    return sentences, embeddings

sentences, embeddings = get_chunk_embeddings(chapter_text)

def get_top_chunk(user_input, sentences, embeddings):
    user_emb = embedding_model.encode([user_input])[0]
    sims = np.dot(embeddings, user_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(user_emb))
    top_idx = np.argmax(sims)
    return sentences[top_idx]

# --- Call OpenAI GPT (new API) ---
def get_ai_response(user_message, chapter, top_chunk=None):
    chapter_context = f"""
    Topic: {chapter['title']}
    Description: {chapter['description']}
    Key vocabulary: {', '.join(chapter['vocabulary'])}
    Grammar focus: {', '.join(chapter['grammar'])}
    Example dialogues: {" | ".join(chapter['sample_dialogues'])}
    """

    system_prompt = f"""
    You are an AI English tutor helping secondary school students practice English conversation.

    You are currently teaching the topic: "{chapter['title']}".

    Your goals:
    1. Lead the conversation proactively — ask engaging, open-ended questions.
    2. Use vocabulary and grammar from this chapter only.
    3. Correct mistakes gently by restating what the student said correctly.
    4. Keep replies short (2–4 sentences) and natural — like a real teacher chatting.
    5. Occasionally ask a quick vocabulary or grammar question related to the topic.
    6. Adapt to the student's level (A2–B2 European CEFR range).
    7. Encourage the student to expand their answers.
    8. Avoid talking about AI or teaching methods — stay fully in character as a teacher.

    {chapter_context}
    """

    # If no user message (i.e. at start), ask the AI to begin
    if not user_message:
        user_message = f"Start the conversation naturally as the teacher for the topic '{chapter['title']}'. Greet the student and ask the first question."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.9,
        max_tokens=300
    )

    ai_reply = response.choices[0].message.content.strip()
    return ai_reply


# --- NEW: Automatically start conversation when chapter is selected ---
if "last_chapter" not in st.session_state or st.session_state.last_chapter != chapter_choice:
    st.session_state.last_chapter = chapter_choice
    st.session_state.chat_history = []
    ai_greeting = get_ai_response("", chapter)
    st.session_state.chat_history.append(("AI", ai_greeting))


# --- Main chat logic ---
if st.button("Send") and user_msg.strip():
    top_chunk = get_top_chunk(user_msg, sentences, embeddings)
    ai_msg = get_ai_response(user_msg, chapter, top_chunk)
    
    st.session_state.chat_history.append(("You", user_msg))
    st.session_state.chat_history.append(("AI", ai_msg))
    
    # Save conversation
    session_log = {"chapter": chapter_choice, "user": user_msg, "ai": ai_msg}
    if not os.path.exists("sessions.json"):
        with open("sessions.json", "w") as f:
            json.dump([session_log], f, indent=2)
    else:
        with open("sessions.json") as f:
            data = json.load(f)
        data.append(session_log)
        with open("sessions.json", "w") as f:
            json.dump(data, f, indent=2)

# --- Display chat history ---
for speaker, msg in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {msg}")
