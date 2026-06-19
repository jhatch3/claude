"""
File streamlit_app.py

A small UI to visualize and exercise the FastAPI routes defined in app.py.

run by (with the FastAPI server already running via `uvicorn app:app --reload`):
>>> streamlit run streamlit_app.py
"""

import requests
import streamlit as st

st.set_page_config(page_title="app.py Route Explorer", page_icon="🔌", layout="wide")

# --- Routes defined in app.py (kept in sync by hand) ---
ROUTES = [
    {
        "method": "GET",
        "path": "/",
        "name": "Health",
        "desc": "Liveness check. Also reports whether Redis is reachable.",
        "color": "#22c55e",  # green
    },
    {
        "method": "POST",
        "path": "/chat",
        "name": "Chat",
        "desc": "Send {user_id, message}. History is stored in Redis per user; "
                "rate limited and prompt-cached.",
        "color": "#3b82f6",  # blue
    },
    {
        "method": "DELETE",
        "path": "/chat/{user_id}",
        "name": "Reset conversation",
        "desc": "Clear a user's stored history so the next /chat starts fresh.",
        "color": "#ef4444",  # red
    },
]

# --- Sidebar config ---
with st.sidebar:
    st.header("Connection")
    base_url = st.text_input("API base URL", value="http://localhost:8000").rstrip("/")
    user_id = st.text_input("user_id", value="demo")
    st.caption("Start the API first: `uvicorn app:app --reload`")


def badge(method: str, color: str) -> str:
    return (
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:6px;font-weight:600;font-size:0.8rem'>{method}</span>"
    )


st.title("🔌 FastAPI Route Explorer")
st.caption("Visualizes and exercises the endpoints in `app.py`.")

# --- Overview of all routes ---
st.subheader("Routes")
for r in ROUTES:
    st.markdown(
        f"{badge(r['method'], r['color'])} &nbsp; `{r['path']}` — **{r['name']}**  \n"
        f"<span style='color:gray'>{r['desc']}</span>",
        unsafe_allow_html=True,
    )

st.divider()

tab_health, tab_chat, tab_reset = st.tabs(
    ["GET / — Health", "POST /chat — Chat", "DELETE /chat/{user_id} — Reset"]
)

# --- GET / ---
with tab_health:
    st.markdown(f"{badge('GET', '#22c55e')} &nbsp; `{base_url}/`", unsafe_allow_html=True)
    if st.button("Send request", key="health_btn"):
        try:
            resp = requests.get(f"{base_url}/", timeout=10)
            ok = resp.json().get("redis", None)
            if ok is True:
                st.success("API up · Redis reachable")
            elif ok is False:
                st.warning("API up · Redis NOT reachable")
            st.json(resp.json())
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")

# --- POST /chat ---
with tab_chat:
    st.markdown(
        f"{badge('POST', '#3b82f6')} &nbsp; `{base_url}/chat` &nbsp; "
        f"payload: `{{\"user_id\": \"{user_id}\", \"message\": \"...\"}}`",
        unsafe_allow_html=True,
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Type a message"):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        try:
            resp = requests.post(
                f"{base_url}/chat",
                json={"user_id": user_id, "message": prompt},
                timeout=60,
            )
            if resp.status_code == 200:
                reply = resp.json()["response"]
            elif resp.status_code == 429:
                reply = f"⏳ Rate limited: {resp.json().get('detail', '')}"
            else:
                reply = f"⚠️ Error {resp.status_code}: {resp.text}"
        except requests.RequestException as e:
            reply = f"⚠️ Request failed: {e}"

        st.session_state.history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

# --- DELETE /chat/{user_id} ---
with tab_reset:
    st.markdown(
        f"{badge('DELETE', '#ef4444')} &nbsp; `{base_url}/chat/{user_id}`",
        unsafe_allow_html=True,
    )
    st.caption(f"Clears server-side history for **{user_id}** in Redis.")
    if st.button("Reset conversation", key="reset_btn", type="primary"):
        try:
            resp = requests.delete(f"{base_url}/chat/{user_id}", timeout=10)
            st.session_state.history = []  # clear the local view too
            st.success("History cleared")
            st.json(resp.json())
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
