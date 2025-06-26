import streamlit as st
import requests

st.title("Nephrology Multi-Agent POC Chat")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("You:", "")
if st.button("Send") and user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    response = requests.post("http://localhost:8000/chat", json={"text": user_input})
    if response.ok:
        data = response.json()
        st.session_state["messages"].append({
            "role": data["agent"],
            "content": data["response"],
            "sources": data.get("sources"),
            "source_type": data.get("source_type")
        })
    else:
        st.session_state["messages"].append({"role": "system", "content": "Error contacting backend."})

for msg in st.session_state["messages"]:
    if msg.get("source_type") == "textbook":
        st.markdown("<span style='color:green'><b>📘 Reference Book Answer</b></span>", unsafe_allow_html=True)
    elif msg.get("source_type") == "web":
        st.markdown("<span style='color:blue'><b>🌐 Web Search Answer</b></span>", unsafe_allow_html=True)
    st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")
    if msg.get("sources"):
        st.markdown("_Sources:_")
        for s in msg["sources"]:
            st.markdown(f"- {s}") 