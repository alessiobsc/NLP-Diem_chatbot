import os
import sys
os.environ.setdefault("PYTHONUNBUFFERED", "1")
from urllib.parse import urlparse
from operator import itemgetter

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from bs4 import BeautifulSoup
import gradio as gr

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_DIR    = "chroma_diem"
COLLECTION    = "diem_knowledge"
SESSION_ID    = "diem-session"
FORCE_REINDEX = "--reindex" in sys.argv

# URL patterns to skip during crawling (static assets, auth pages, etc.)
EXCLUDE_DIRS = [
    "/rescue/css/", "/rescue/js/", "/rescue/assets/",
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".ico", ".woff", ".woff2", ".ttf", ".eot",
    "/idp/", "/password-recovery", "/login",
]

# ─────────────────────────────────────────────────────────────────────────────
# API Key
# ─────────────────────────────────────────────────────────────────────────────
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if api_key:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    print("API Key loaded.")
else:
    print("ERROR: HUGGINGFACEHUB_API_TOKEN not found.")

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
chat_model = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.1
))

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

# ─────────────────────────────────────────────────────────────────────────────
# HTML Extractor
# Keeps heading tags (h1/h2/h3) so HTMLSectionSplitter can split on them.
# Removes noise: nav, footer, scripts, ads.
# ─────────────────────────────────────────────────────────────────────────────
def html_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "aside", "iframe"]):
        tag.decompose()
    for selector in ["main", "article", "#content", ".content",
                     "#main", ".main-content", ".entry-content"]:
        content = soup.select_one(selector)
        if content:
            return str(content)
    body = soup.find("body")
    return str(body) if body else html


# ─────────────────────────────────────────────────────────────────────────────
# Utility: extract links to docenti/corsi from raw HTML docs
# ─────────────────────────────────────────────────────────────────────────────
def extract_external_links(raw_docs: list) -> dict:
    targets = {"docenti.unisa.it": set(), "corsi.unisa.it": set()}
    for doc in raw_docs:
        try:
            soup = BeautifulSoup(doc.page_content, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href.startswith("http"):
                    continue
                for domain in targets:
                    if domain in href:
                        clean = href.split("?")[0].split("#")[0].rstrip("/")
                        targets[domain].add(clean)
        except Exception:
            pass
    return targets


def get_section_base(url: str) -> str:
    """Return the first-segment root of a URL path.

    https://docenti.unisa.it/003145/home  ->  https://docenti.unisa.it/003145/
    https://corsi.unisa.it/ing-inf/home   ->  https://corsi.unisa.it/ing-inf/
    """
    parsed = urlparse(url)
    parts  = parsed.path.strip("/").split("/")
    base_path = f"/{parts[0]}/" if parts and parts[0] else "/"
    return f"{parsed.scheme}://{parsed.netloc}{base_path}"


# ─────────────────────────────────────────────────────────────────────────────
# Crawl helper
# ─────────────────────────────────────────────────────────────────────────────
def crawl(start_url: str, base_url: str, max_depth: int = 2) -> list:
    try:
        loader = RecursiveUrlLoader(
            start_url,
            base_url=base_url,
            max_depth=max_depth,
            prevent_outside=True,
            timeout=15,
            check_response_status=True,
            extractor=html_extractor,
            exclude_dirs=EXCLUDE_DIRS,
        )
        return loader.load()
    except Exception as e:
        print(f"  FAILED {start_url}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: crawl -> chunk -> embed -> store
# ─────────────────────────────────────────────────────────────────────────────
def build_index() -> Chroma:

    # ── PHASE 1: Load ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1 – Loading web pages")
    print("=" * 60)

    all_docs = []

    # 1a. Crawl diem.unisa.it WITHOUT extractor to keep raw HTML for link extraction
    print("\n[1/3] Crawling www.diem.unisa.it ...")
    raw_loader = RecursiveUrlLoader(
        "https://www.diem.unisa.it/",
        base_url="https://www.diem.unisa.it/",
        max_depth=3,
        prevent_outside=True,
        timeout=15,
        check_response_status=True,
        exclude_dirs=EXCLUDE_DIRS,
    )
    raw_diem = raw_loader.load()
    print(f"  -> {len(raw_diem)} pages found")

    # Extract links before cleaning
    external    = extract_external_links(raw_diem)
    docenti_urls = list(external["docenti.unisa.it"])
    corsi_urls   = list(external["corsi.unisa.it"])
    print(f"  -> {len(docenti_urls)} docenti links, {len(corsi_urls)} corsi links")

    # Clean diem docs in-place (now HTML extractor strips noise but keeps headings)
    for doc in raw_diem:
        doc.page_content = html_extractor(doc.page_content)
    all_docs.extend(raw_diem)

    # 1b. Crawl docenti.unisa.it  (cap at 50 for prototype)
    cap_docenti = min(50, len(docenti_urls))
    print(f"\n[2/3] Crawling docenti.unisa.it ({cap_docenti} faculty pages) ...")
    for i, url in enumerate(docenti_urls[:cap_docenti], 1):
        base = get_section_base(url)
        docs = crawl(url, base_url=base, max_depth=2)
        all_docs.extend(docs)
        print(f"  [{i:02d}/{cap_docenti}] {url}  ({len(docs)} sub-pages)")

    # 1c. Crawl corsi.unisa.it  (cap at 30 for prototype)
    cap_corsi = min(30, len(corsi_urls))
    print(f"\n[3/3] Crawling corsi.unisa.it ({cap_corsi} course pages) ...")
    for i, url in enumerate(corsi_urls[:cap_corsi], 1):
        base = get_section_base(url)
        docs = crawl(url, base_url=base, max_depth=2)
        all_docs.extend(docs)
        print(f"  [{i:02d}/{cap_corsi}] {url}  ({len(docs)} sub-pages)")

    print(f"\nTotal documents loaded: {len(all_docs)}")

    # ── PHASE 2: Chunk ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 – Chunking")
    print("=" * 60)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    )
    chunks = char_splitter.split_documents(all_docs)
    print(f"  -> {len(chunks)} chunks from {len(all_docs)} documents")

    # ── PHASE 3: Embed & Index ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 – Embedding and indexing")
    print("=" * 60)

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        vectorstore.add_documents(chunks[i : i + batch_size])
        print(f"  -> {min(i + batch_size, len(chunks))}/{len(chunks)} chunks indexed", flush=True)

    print("\nIndexing complete.")
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Load existing index or build from scratch
# ─────────────────────────────────────────────────────────────────────────────
db_file = os.path.join(CHROMA_DIR, "chroma.sqlite3")

if FORCE_REINDEX or not os.path.exists(db_file):
    vectorstore = build_index()
else:
    print("Loading existing Chroma index...")
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    try:
        print(f"  -> {vectorstore._collection.count()} chunks in index")
    except Exception:
        print("  -> Index loaded")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ─────────────────────────────────────────────────────────────────────────────
# RAG Chain  (mirrors the exercise notebook structure)
# ─────────────────────────────────────────────────────────────────────────────
rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for the DIEM department "
     "(Department of Information and Electrical Engineering and Applied Mathematics) "
     "at the University of Salerno, Italy. "
     "Answer questions using ONLY the provided context. Do not use prior knowledge. "
     "If the answer is not in the context, say: "
     "'I don't have that information in my knowledge base.' "
     "If the question is unrelated to DIEM or the University of Salerno, say: "
     "'This question is outside my scope. I can only answer questions about DIEM.'"
    ),
    ("placeholder", "{history}"),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"),
])

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's question into a self-contained search query using the chat history. "
     "Resolve all pronouns and references (e.g. 'it', 'they', 'that professor'). "
     "Return ONLY the rewritten query."
    ),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])

rewrite_chain = (
    rewrite_prompt
    | chat_model
    | RunnableLambda(lambda m: m.content.strip())
)

rag_chain = (
    {
        "docs":     itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "history":  itemgetter("history"),
    }
    | RunnableLambda(lambda x: {
        **x,
        "context": "\n\n---\n\n".join(d.page_content for d in x["docs"]),
    })
    | RunnableLambda(lambda x: {
        "answer": chat_model.invoke(
            rag_prompt.invoke({
                "context":  x["context"],
                "question": x["question"],
                "history":  x["history"],
            })
        ).content,
        "sources": x["docs"],
    })
)

rag_chain_with_rewrite = (
    RunnablePassthrough()
    | RunnablePassthrough.assign(question=rewrite_chain)
    | rag_chain
)

store: dict = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain_with_rewrite,
    get_history,
    input_messages_key="question",
    output_messages_key="answer",
    history_messages_key="history",
)

# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
def chat_fn(message: str, history: list) -> str:
    result = conversational_rag.invoke(
        {"question": message},
        config={"configurable": {"session_id": SESSION_ID}},
    )
    answer = result["answer"]

    # Append unique source URLs at the bottom of the answer
    sources = result.get("sources", [])
    if sources:
        seen, unique = set(), []
        for doc in sources:
            url = doc.metadata.get("source", "")
            if url and url not in seen:
                seen.add(url)
                unique.append(url)
        if unique:
            answer += "\n\n**Sources:**\n" + "\n".join(f"- {u}" for u in unique)

    return answer


demo = gr.ChatInterface(
    fn=chat_fn,
    title="DIEM Chatbot",
    description=(
        "Ask questions about the DIEM department (University of Salerno): "
        "degree programs, faculty, research, courses, regulations, and more."
    ),
    examples=[
        "What degree programs are offered by DIEM?",
        "Where is DIEM located?",
        "What research areas are active at DIEM?",
        "Who is responsible for internationalization at DIEM?",
        "Which laboratories are available at DIEM?",
    ],
    chatbot=gr.Chatbot(height=500),
)

if __name__ == "__main__":
    demo.launch()
