import uuid

import streamlit as st

import backend
from metadata_utils import get_file_metadata

st.set_page_config(page_title="PRISMA AI", layout="wide")

# --- UI Polish: Gradient Header & Subtitle ---
st.markdown("""
<div style="text-align: center; margin-top: -2rem; padding-bottom: 2rem;">
    <h1 style="background: -webkit-linear-gradient(45deg, #60A5FA, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 4rem; margin-bottom: 0;">PRISMA AI</h1>
    <p style="color: #9CA3AF; font-size: 1.15rem; font-style: italic; margin-top: -1rem; padding-left: 1.2rem">Chat With Anything You Upload</p>
</div>
""", unsafe_allow_html=True)

# --- CSS for animations, badges, and truncation ---
st.markdown("""
<style>
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 3px solid #444;
    border-top: 3px solid #4CAF50;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 8px;
    vertical-align: middle;
}
/* File name truncation */
.file-name-truncate {
    display: inline-block;
    max-width: 130px; 
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    vertical-align: middle;
}
/* Chunk badge pill */
.chunk-badge {
    background-color: #374151;
    color: #D1D5DB;
    padding: 2px 12px;
    border-radius: 12px;
    font-size: 0.75rem;
    margin-left: 6px;
    vertical-align: middle;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)


# --- Cache the heavy models at the Streamlit layer ---
# backend.get_whisper_model / get_embedder already cache themselves with
# lru_cache, but wrapping them in st.cache_resource too keeps Streamlit's
# own re-run behavior from re-triggering any loading logic.
@st.cache_resource
def load_whisper():
    return backend.get_whisper_model()


@st.cache_resource
def load_embedder():
    return backend.get_embedder()


load_whisper()
load_embedder()


def new_session():
    st.session_state.current_session_id = str(uuid.uuid4())[:8]
    st.session_state.messages = []


if "current_session_id" not in st.session_state:
    new_session()

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Sidebar: uploads
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("**Upload Files**")

    groq_ready = backend.get_groq_client() is not None
    if not groq_ready:
        st.warning("GROQ_API_KEY is not set. Questions won't work until it's configured.")

    uploaded_files = st.file_uploader(
        "Upload video, audio, PDF, DOCX, PPTX, or TXT",
        accept_multiple_files=True,
        type=["mp4", "webm", "mp3", "wav", "pdf", "docx", "pptx", "txt"],
    )

    if uploaded_files and st.button("Process Files"):
        

        status_placeholders = {}
        for uf in uploaded_files:
            status_placeholders[uf.name] = st.empty()
            status_placeholders[uf.name].markdown(f":material/hourglass_empty: **{uf.name}** — waiting in queue...")

        any_success = False
        for uf in uploaded_files:
            status_placeholders[uf.name].markdown(
                f'<span class="spinner"></span> **{uf.name}** — processing...',
                unsafe_allow_html=True,
            )

            path = backend.save_uploaded_file(uf)
            title, error = backend.process_file(path, uf.name)
            if error:
                status_placeholders[uf.name].markdown(f":material/error: **{uf.name}** — {error}")
            else:
                status_placeholders[uf.name].markdown(f":material/check_circle: **{title}** — done")
                any_success = True

        if any_success:
            with st.spinner("Updating vector database..."):
                ok = backend.rebuild_vector_store()
            if ok:
                st.success("Files processed successfully. You can now ask questions.")
            else:
                st.error("No usable content found to index.")
        else:
            st.error("No files were processed successfully.")
    
    
    st.divider()

    file_count, chunk_count, file_list = backend.get_knowledge_base_stats()

    with st.expander(f":material/library_books: Knowledge Base ({file_count} files)"):
        st.write(f"**Total Files:** {file_count}")
        st.write(f"**Total Chunks:** {chunk_count}")

        st.divider()

        # --- Manage Files ---
        with st.expander(":material/description: Manage Files", expanded=True):
            if not file_list:
                st.caption("No files uploaded yet.")
            else:
                for f in file_list:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(
                            f"""<div style="margin-top: 8px;">
                                 <span class="file-name-truncate" title="{f['title']}">{f['title']}</span> 
                                <span class="chunk-badge">{f['chunk_count']} Chunks</span>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    with col2:
                        # type="tertiary" removes the bounding box for a cleaner icon
                        if st.button(":material/delete:", key=f"del_{f['file_id']}", type="tertiary", use_container_width=True):
                            backend.delete_file(f['file_id'])
                            backend.rebuild_vector_store()
                            st.rerun()

        st.divider()

        # --- Clear Knowledge Base, with confirmation ---
        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = False

        if not st.session_state.confirm_clear:
            if st.button(":material/delete_forever: Clear Knowledge Base", type="primary", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("This will permanently delete ALL uploaded files. Are you sure?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(":material/check: Yes, delete", use_container_width=True):
                    backend.clear_knowledge_base()
                    st.session_state.confirm_clear = False
                    st.rerun()
            with col2:
                if st.button(":material/cancel: Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()
    
    st.divider()
    if st.button(":material/add: New Chat", use_container_width=True):
        new_session()
        st.rerun()

    st.subheader("**Chat History**")
    chat_search = st.text_input(
        "Chats", 
        placeholder="Search Chats...", 
        key="chat_search", 
        label_visibility="collapsed"
    )

    sessions = backend.load_all_sessions()
    if chat_search:
        sessions = [s for s in sessions if chat_search.lower() in s["title"].lower()]

    for session in sessions:
        # Create columns for the chat session button and delete button
        col1, col2 = st.columns([5, 1])
        
        label = session["title"] + ("..." if len(session["title"]) >= 50 else "")
        is_current = session["session_id"] == st.session_state.current_session_id
        button_label = f":material/arrow_right: {label}" if is_current else f" {label}"

        with col1:
            # Added use_container_width=True to make the buttons align nicely
            if st.button(button_label, key=f"load_{session['session_id']}", use_container_width=True):
                st.session_state.current_session_id = session["session_id"]
                st.session_state.messages = session["messages"]
                st.rerun()
                
        with col2:
            if st.button(":material/delete:", key=f"del_chat_{session['session_id']}", type="tertiary", use_container_width=True):
    
                backend.delete_session(session["session_id"]) 
                
                # If the user deletes the chat they are currently looking at, reset to a new session
                if is_current:
                    new_session()
                    
                st.rerun()


# ---------------------------------------------------------------------------
# Main: chat interface
# ---------------------------------------------------------------------------

if backend.has_index() and backend.is_index_stale():
    st.caption("Your indexed files were built with an older version of the app — click **Process Files** in the sidebar to rebuild.")

query = st.chat_input("Ask about your uploaded files...")
if query:
    answer, sources = backend.answer_query(query)
    st.session_state.messages.append((query, answer, sources))
    backend.save_session(st.session_state.current_session_id, st.session_state.messages)

for q, a, sources in st.session_state.messages:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        if sources:
            with st.expander("Sources"):
                seen_titles = {}
                for s in sources:
                    seen_titles.setdefault(s["title"], []).append(s)

                for src_title, chunks_for_title in seen_titles.items():
                    if chunks_for_title[0]["start"] is not None:
                        for c in chunks_for_title:
                            st.write(f" **{src_title}** — {round(c['start'], 1)}s to {round(c['end'], 1)}s")
                    else:
                        count = len(chunks_for_title)
                        label = f"({count} matching sections)" if count > 1 else ""
                        st.write(f" **{src_title}** {label}")