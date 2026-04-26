import os
import io
import re
import zipfile
from typing import List, Tuple
from pathlib import Path
from xml.etree import ElementTree as ET

import streamlit as st
from dotenv import load_dotenv

from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


# ----------------------------
# Page & Theme
# ----------------------------
st.set_page_config(
    page_title="Employment Stability Paper RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
<style>
  .main { background: #f6fbf9; }
  .app-title {
    background: linear-gradient(90deg, #00704a, #2d8659);
    color: white; padding: 16px 24px; border-radius: 8px; font-weight: 700;
    display:flex; align-items:center; gap:10px; margin-bottom:16px;
  }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; background:#e8f5f0; color:#00704a; font-size:12px; margin-left:8px;}
  .card { background:white; padding:16px; border-radius:8px; border-left:4px solid #00704a; box-shadow:0 2px 6px rgba(0,0,0,0.06); color:#00704a; }
  .hint { color:#2d8659; font-size:13px; }
  .footer { text-align:center; color:#2d8659; margin-top:24px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="app-title">
  <span style="font-size:26px">Employment Stability and Corporate Cash Holdings RAG Chatbot</span>
  <span class="pill">Academic Paper · RAG · FAISS · OpenAI</span>
</div>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Helpers
# ----------------------------
def load_api_key() -> str:
    load_dotenv(override=False)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
        api_key = st.session_state.openai_api_key
    return api_key


def load_deploy_link() -> str:
    """Load deployed app URL from env or DEPLOY_LINK.txt at repo root."""
    link = os.getenv("STREAMLIT_DEPLOY_URL") or os.getenv("DEPLOY_URL") or ""
    if link:
        return link.strip()
    try:
        path = Path(__file__).resolve().parent.parent / "DEPLOY_LINK.txt"
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            return content
    except Exception:
        pass
    return ""


def extract_docx_text(raw_bytes: bytes) -> str:
    """Extract readable text from a DOCX file using only the standard library."""
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        xml_bytes = zf.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        runs = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        if runs:
            paragraphs.append("".join(runs))
    return "\n".join(paragraphs)


def extract_legacy_doc_text(raw_bytes: bytes) -> str:
    """
    Best-effort extraction for old OLE .doc files.
    This handles legacy files whose readable text is visible as UTF-16LE.
    """
    decoded = raw_bytes.decode("utf-16le", errors="ignore")
    candidates = re.findall(r"[가-힣A-Za-z0-9()\[\]{}.,;:!?%/ㆍ·\-~\s]{8,}", decoded)
    text = "\n".join(part.strip() for part in candidates if part.strip())
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_uploaded_file(file) -> Tuple[str, str]:
    """
    Returns (text, source_name)
    Supports PDF, TXT, DOCX, and best-effort legacy DOC.
    """
    filename = file.name
    name_lower = filename.lower()
    raw_bytes = file.read()

    if name_lower.endswith(".pdf"):
        try:
            laparams = LAParams(
                line_margin=0.2,
                char_margin=2.0,
                word_margin=0.1,
                boxes_flow=None,
            )
            text = extract_text(io.BytesIO(raw_bytes), laparams=laparams)
        except Exception as e:
            raise RuntimeError(f"PDF 파싱 실패: {e}")
    elif name_lower.endswith(".txt"):
        try:
            text = raw_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = raw_bytes.decode("cp949", errors="ignore")
    elif name_lower.endswith(".docx"):
        try:
            text = extract_docx_text(raw_bytes)
        except Exception as e:
            raise RuntimeError(f"DOCX 파싱 실패: {e}")
    elif name_lower.endswith(".doc"):
        text = extract_legacy_doc_text(raw_bytes)
        if len(text) < 200:
            raise RuntimeError(
                "DOC 파일에서 충분한 텍스트를 추출하지 못했습니다. PDF 또는 TXT로 변환 후 업로드하세요."
            )
    else:
        raise RuntimeError("지원하지 않는 파일 형식입니다. PDF, TXT, DOC, DOCX 파일을 업로드하세요.")

    text = text.strip()
    if not text:
        raise RuntimeError("파일에서 텍스트를 추출하지 못했습니다.")
    return text, filename


def chunk_documents(texts_with_sources: List[Tuple[str, str]], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    docs: List[Document] = []
    for text, source in texts_with_sources:
        for chunk_number, chunk in enumerate(splitter.split_text(text), start=1):
            docs.append(Document(page_content=chunk, metadata={"source": source, "chunk_number": chunk_number}))
    return docs


def build_vectorstore(docs: List[Document], api_key: str) -> FAISS:
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model="text-embedding-3-small",
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )
    vs = FAISS.from_documents(docs, embeddings)
    return vs


def format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        chunk_number = d.metadata.get("chunk_number", i)
        blocks.append(f"[Document {i}] (Source: {src}, Chunk: {chunk_number})\n{d.page_content}")
    return "\n\n".join(blocks)


def generate_answer(
    query: str,
    retrieved_docs: List[Document],
    api_key: str,
    model: str,
    temperature: float,
    question_type: str,
) -> str:
    context_text = format_context(retrieved_docs)
    system_prompt = """You are an academic research assistant specializing in economics and corporate finance papers.
You answer only based on the uploaded paper: “Employment stability and corporate cash holdings” by Han and Kim (2021).

Follow these rules strictly:
1) Use only the uploaded paper as the evidence source.
2) Do not invent findings, hypotheses, coefficients, or interpretations.
3) When answering empirical questions, mention the relevant section, table, or variable if available.
4) Explain academic concepts in a clear way for students and researchers.
5) For regression results, distinguish between the dependent variable, independent variable, control variables, interaction terms, fixed effects, and interpretation.
6) If the paper does not provide enough information, say that the paper does not provide sufficient evidence.
7) Keep answers structured, concise, and suitable for seminar preparation."""

    user_prompt = (
        "Question Type: " + question_type + "\n\n"
        "Question:\n" + query + "\n\n"
        "Use the retrieved paper context below and answer in the required academic format.\n"
        "Answer in the same language as the user's question unless the user asks otherwise.\n\n"
        "Required academic format:\n\n"
        "[Short Answer]\n"
        "- Direct answer to the user's question\n\n"
        "[Evidence from the Paper]\n"
        "- Relevant section, table, or variable from the paper\n\n"
        "[Detailed Explanation]\n"
        "- Explain the logic, theory, data, or empirical result\n\n"
        "[Interpretation]\n"
        "- What the result means academically or practically\n\n"
        "[Limitations / Caution]\n"
        "- Mention missing information, assumptions, or cautions if relevant\n\n"
        "Context:\n" + context_text
    )

    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    res = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    return res.content if hasattr(res, "content") else str(res)


# ----------------------------
# Sidebar Controls
# ----------------------------
with st.sidebar:
    st.markdown("### Settings")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="환경변수 OPENAI_API_KEY 또는 여기 입력 (저장 안 됨)",
    )
    if api_key_input:
        st.session_state.openai_api_key = api_key_input

    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    question_type = st.selectbox(
        "Question Type",
        ["Auto", "Summary", "Theory", "Hypothesis", "Data", "Variables", "Regression Results", "Robustness", "Contribution", "Limitations"],
        index=0,
    )
    chunk_size = st.slider("Chunk Size", 200, 2000, 900, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 400, 180, 10)
    top_k = st.slider("Retrieved Chunks (k)", 1, 10, 5, 1)

    st.markdown("---")
    clear_btn = st.button("🗑️ Clear Paper Index / Chat")


# ----------------------------
# Session State
# ----------------------------
if clear_btn:
    for key in ["vectorstore", "docs_info", "messages"]:
        if key in st.session_state:
            del st.session_state[key]

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of (role, content)


# ----------------------------
# Upload & Index Build
# ----------------------------
st.markdown("#### 1) Upload Paper")
st.markdown(
    '<div class="card">Upload the Han and Kim (2021) Paper, then build a dedicated academic paper vector index.</div>',
    unsafe_allow_html=True,
)

uploads = st.file_uploader(
    "Upload Paper",
    type=["pdf", "txt", "doc", "docx"],
    accept_multiple_files=True,
)
st.caption("PDF/TXT/DOCX are recommended. Legacy DOC extraction quality can vary by document structure.")

api_key = load_api_key()

build_col1, build_col2 = st.columns([1, 3])
with build_col1:
    build_clicked = st.button("🧠 Build Paper Index")
with build_col2:
    st.markdown("<span class=hint>After uploading the paper, build the index before asking questions.</span>", unsafe_allow_html=True)

if build_clicked:
    if not api_key:
        st.error("OpenAI API Key is required. Enter it in the sidebar or set the environment variable.")
    elif not uploads:
        st.error("Upload at least one file.")
    else:
        try:
            texts_with_sources: List[Tuple[str, str]] = []
            for f in uploads:
                text, src = read_uploaded_file(f)
                texts_with_sources.append((text, src))

            docs = chunk_documents(texts_with_sources, chunk_size, chunk_overlap)
            vs = build_vectorstore(docs, api_key)
            st.session_state.vectorstore = vs
            st.session_state.docs_info = {
                "num_files": len(uploads),
                "num_chunks": len(docs),
                "files": [f.name for f in uploads],
            }
            st.success("Paper index built. Ask questions about the Han and Kim (2021) Paper.")
        except Exception as e:
            st.error(f"Index build failed: {e}")


# ----------------------------
# Recommended Questions
# ----------------------------
st.markdown("#### Recommended Questions")
demo_questions = [
    "What is the main research question of this paper?",
    "How do the authors measure employment stability?",
    "What is the dependent variable?",
    "What does Table 1 show?",
    "How should Table 2 be interpreted?",
    "Why do R&D intensive firms matter in this paper?",
    "What is firm-specific knowledge?",
    "What robustness checks do the authors perform?",
    "What is the main contribution of this paper?",
    "What are the limitations of this study?",
]
question_cols = st.columns(2)
for i, sample_question in enumerate(demo_questions):
    with question_cols[i % 2]:
        if st.button(sample_question, key=f"demo_question_{i}", use_container_width=True):
            st.session_state.pending_question = sample_question


# ----------------------------
# Index Status
# ----------------------------
st.markdown("#### 2) Paper Index Status")
status_container = st.container()
with status_container:
    if "vectorstore" in st.session_state:
        info = st.session_state.get("docs_info", {})
        st.markdown(
            f"- Index status: ✅ Ready  "+
            f"- Files: {info.get('num_files', 0)}  "+
            f"- Chunks: {info.get('num_chunks', 0)}"
        )
        if info.get("files"):
            st.caption("Uploaded files: " + ", ".join(info["files"]))
    else:
        st.markdown("- Index status: ⏳ Not built yet")


# ----------------------------
# Chat Interface
# ----------------------------
st.markdown("#### 3) Answer based on Han and Kim (2021)")
chat_container = st.container()

with chat_container:
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    pending_question = st.session_state.pop("pending_question", None)
    typed_question = st.chat_input("Ask about the Han and Kim (2021) Paper...")
    user_input = pending_question or typed_question

    if user_input:
        if "vectorstore" not in st.session_state:
            st.error("Upload the paper and build the index first.")
        elif not api_key:
            st.error("OpenAI API Key is required.")
        else:
            st.session_state.messages.append(("user", user_input))
            with st.chat_message("user"):
                st.markdown(user_input)

            try:
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k},
                )
                # LangChain retrievers use .invoke() in recent versions
                retrieved_docs: List[Document] = retriever.invoke(user_input)

                with st.spinner("Generating answer..."):
                    answer = generate_answer(user_input, retrieved_docs, api_key, model, temperature, question_type)

                with st.chat_message("assistant"):
                    st.caption("Answer based on Han and Kim (2021)")
                    st.markdown(answer)
                    with st.expander("View Retrieved Paper Evidence"):
                        for i, d in enumerate(retrieved_docs, start=1):
                            src = d.metadata.get("source", "unknown")
                            chunk_number = d.metadata.get("chunk_number", i)
                            preview = d.page_content[:1000] + ("…" if len(d.page_content) > 1000 else "")
                            st.markdown(f"**Retrieved Evidence {i}**")
                            st.markdown(f"- Source file name: `{src}`")
                            st.markdown(f"- Retrieved chunk number: `{chunk_number}`")
                            st.write(preview)

                st.session_state.messages.append(("assistant", answer))
            except Exception as e:
                err_msg = f"An error occurred: {e}"
                with st.chat_message("assistant"):
                    st.error(err_msg)
                st.session_state.messages.append(("assistant", err_msg))


st.markdown("---")
st.markdown(
    "<div class=footer>© Han and Kim (2021) Paper RAG Chatbot · Streamlit · LangChain · FAISS</div>",
    unsafe_allow_html=True,
)


