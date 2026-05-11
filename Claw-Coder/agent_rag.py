"""
Standalone agent with web tools, terminal tools, PDF RAG, and Tree-sitter code RAG.

This file combines the useful parts of:
- agent.py: chat loop, web search, browser opening, terminal execution
- clock.py: PDF loading, chunking, ChromaDB storage, Ollama embeddings
- clock_tree_rag.py: multi-language Tree-sitter code chunking

Setup:
    pip install chromadb ollama ddgs pypdf tree-sitter tree-sitter-python
    pip install tree-sitter-javascript tree-sitter-typescript tree-sitter-json
    pip install tree-sitter-html tree-sitter-css tree-sitter-java tree-sitter-go tree-sitter-rust
    ollama serve
    ollama pull qwen3-embedding:4b
    ollama pull granite4.1:8b

Examples:
    python agent_rag.py languages
    python agent_rag.py code-chunks agent.py
    python agent_rag.py ingest-code agent.py
    python agent_rag.py ingest-pdf data/2509.24435v1.pdf
    python agent_rag.py search-kb "where is execute_tool?"
    python agent_rag.py chat
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import re
import shlex
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import ollama
from ddgs import DDGS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("agent_rag.log"), logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


DEFAULT_CHAT_MODEL = "llama3.2:3b"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:4b"
DEFAULT_DB_PATH = "agent_rag_chroma_db"
DEFAULT_COLLECTION = "agent_mixed_knowledge"
DEFAULT_PDF = Path(__file__).resolve().parent /"data" / "2509.24435v1.pdf" # you can place any document you desire for it to ingest then run python <file> ingest


LANGUAGE_SPECS: Dict[str, Dict[str, str]] = {
    "python": {"module": "tree_sitter_python", "function": "language"},
    "javascript": {"module": "tree_sitter_javascript", "function": "language"},
    "typescript": {"module": "tree_sitter_typescript", "function": "language_typescript"},
    "tsx": {"module": "tree_sitter_typescript", "function": "language_tsx"},
    "json": {"module": "tree_sitter_json", "function": "language"},
    "html": {"module": "tree_sitter_html", "function": "language"},
    "css": {"module": "tree_sitter_css", "function": "language"},
    "java": {"module": "tree_sitter_java", "function": "language"},
    "go": {"module": "tree_sitter_go", "function": "language"},
    "rust": {"module": "tree_sitter_rust", "function": "language"},
}


EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".json": "json",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}


@dataclass(slots=True)
class Document:
    page_content: str
    metadata: Dict[str, Any]


@dataclass(slots=True)
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float]


def require_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("ChromaDB is missing. Install it with: pip install chromadb") from exc
    return chromadb


def require_pdf_reader():
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is missing. Install it with: pip install pypdf") from exc
    return PdfReader


def require_tree_sitter():
    try:
        from tree_sitter import Language, Parser, Query, QueryCursor
    except ImportError as exc:
        raise RuntimeError("Tree-sitter is missing. Install it with: pip install tree-sitter") from exc
    return Language, Parser, Query, QueryCursor


def available_languages() -> Dict[str, Dict[str, Any]]:
    status: Dict[str, Dict[str, Any]] = {}
    for language, spec in LANGUAGE_SPECS.items():
        try:
            module = importlib.import_module(spec["module"])
            getattr(module, spec["function"])
            status[language] = {"available": True, "module": spec["module"], "install": None}
        except Exception:
            package = spec["module"].replace("_", "-")
            status[language] = {
                "available": False,
                "module": spec["module"],
                "install": f"pip install {package}",
            }
    return status


def infer_language(path: str) -> Optional[str]:
    return EXTENSION_TO_LANGUAGE.get(Path(path).suffix.lower())


def load_tree_sitter_language(language_name: str):
    Language, Parser, Query, QueryCursor = require_tree_sitter()
    spec = LANGUAGE_SPECS.get(language_name)
    if not spec:
        supported = ", ".join(sorted(LANGUAGE_SPECS))
        raise RuntimeError(f"Unsupported language '{language_name}'. Supported: {supported}")

    try:
        module = importlib.import_module(spec["module"])
        language_fn = getattr(module, spec["function"])
    except Exception as exc:
        package = spec["module"].replace("_", "-")
        raise RuntimeError(
            f"Tree-sitter grammar for {language_name} is missing. Install it with: "
            f"pip install {package}"
        ) from exc

    parser = Parser()
    parser.language = Language(language_fn())
    return parser, Query, QueryCursor, parser.language


def node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def load_pdf(path: str) -> List[Document]:
    pdf_path = Path(path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    PdfReader = require_pdf_reader()
    pdf_reader = PdfReader(str(pdf_path))
    docs: List[Document] = []
    for index, page in enumerate(pdf_reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(pdf_path), "page": index, "kind": "pdf"},
                )
            )
    return docs


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 250,
) -> List[Document]:
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    chunks: List[Document] = []
    step = chunk_size - chunk_overlap
    for doc in documents:
        text = " ".join(doc.page_content.split())
        for start in range(0, len(text), step):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                metadata = dict(doc.metadata)
                metadata.update({"chunk_start": start, "chunk_end": end})
                chunks.append(Document(page_content=chunk_text, metadata=metadata))
            if end >= len(text):
                break
    return chunks


def fallback_code_chunks(path: str, text: str, language: str, chunk_size: int = 1200) -> List[Document]:
    chunks: List[Document] = []
    for index, start in enumerate(range(0, len(text), chunk_size)):
        end = min(start + chunk_size, len(text))
        content = text[start:end].strip()
        if content:
            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(Path(path).resolve()),
                        "kind": "code",
                        "language": language,
                        "symbol_type": "text_chunk",
                        "symbol_name": f"chunk_{index}",
                        "start_byte": start,
                        "end_byte": end,
                    },
                )
            )
    return chunks


def query_for_language(language: str) -> Optional[str]:
    if language == "python":
        return """
        (function_definition name: (identifier) @name) @definition.function
        (class_definition name: (identifier) @name) @definition.class
        """
    if language == "javascript":
        return """
        (function_declaration name: (identifier) @name) @definition.function
        (class_declaration name: (identifier) @name) @definition.class
        (method_definition name: (property_identifier) @name) @definition.method
        """
    if language in {"typescript", "tsx"}:
        return """
        (function_declaration name: (identifier) @name) @definition.function
        (class_declaration name: (type_identifier) @name) @definition.class
        (method_definition name: (property_identifier) @name) @definition.method
        """
    if language == "java":
        return """
        (class_declaration name: (identifier) @name) @definition.class
        (method_declaration name: (identifier) @name) @definition.method
        """
    if language == "go":
        return """
        (function_declaration name: (identifier) @name) @definition.function
        (method_declaration name: (field_identifier) @name) @definition.method
        (type_declaration (type_spec name: (type_identifier) @name)) @definition.type
        """
    if language == "rust":
        return """
        (function_item name: (identifier) @name) @definition.function
        (struct_item name: (type_identifier) @name) @definition.struct
        (enum_item name: (type_identifier) @name) @definition.enum
        (impl_item) @definition.impl
        """
    return None


def tree_sitter_code_chunks(path: str, language: Optional[str] = None) -> List[Document]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Code file not found: {file_path}")

    detected_language = language or infer_language(str(file_path))
    if not detected_language:
        raise RuntimeError(f"Could not infer language for file: {file_path}")

    text = file_path.read_text(encoding="utf-8", errors="replace")
    source = text.encode("utf-8")
    parser, Query, QueryCursor, ts_language = load_tree_sitter_language(detected_language)
    tree = parser.parse(source)

    query_text = query_for_language(detected_language)
    if not query_text:
        return fallback_code_chunks(str(file_path), text, detected_language)

    captures = QueryCursor(Query(ts_language, query_text)).captures(tree.root_node)
    name_by_position: Dict[Tuple[int, int], str] = {}
    for name_node in captures.get("name", []):
        name_by_position[(name_node.start_byte, name_node.end_byte)] = node_text(source, name_node)

    chunks: List[Document] = []
    for capture_name, nodes in captures.items():
        if capture_name == "name":
            continue
        for node in nodes:
            symbol_name = "anonymous"
            for child in node.children:
                key = (child.start_byte, child.end_byte)
                if key in name_by_position:
                    symbol_name = name_by_position[key]
                    break
            content = node_text(source, node).strip()
            if not content:
                continue
            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "kind": "code",
                        "language": detected_language,
                        "symbol_type": capture_name.replace("definition.", ""),
                        "symbol_name": symbol_name,
                        "start_byte": node.start_byte,
                        "end_byte": node.end_byte,
                        "start_point": list(node.start_point),
                        "end_point": list(node.end_point),
                        "has_error": bool(tree.root_node.has_error),
                    },
                )
            )

    return chunks or fallback_code_chunks(str(file_path), text, detected_language)


def stable_id(document: Document) -> str:
    source = document.metadata.get("source", "unknown")
    kind = document.metadata.get("kind", "unknown")
    start = document.metadata.get("chunk_start", document.metadata.get("start_byte", 0))
    name = document.metadata.get("symbol_name", "")
    digest = hashlib.sha256(
        f"{source}:{kind}:{start}:{name}:{document.page_content}".encode("utf-8")
    ).hexdigest()
    return digest[:24]


def ollama_embed(texts: Iterable[str], model: str = DEFAULT_EMBEDDING_MODEL) -> List[List[float]]:
    values = list(texts)
    if not values:
        return []
    try:
        response = ollama.embed(model=model, input=values)
    except Exception as exc:
        raise RuntimeError(
            "Ollama embedding failed. Make sure Ollama is running and the embedding "
            f"model is pulled: ollama pull {model}"
        ) from exc

    embeddings = response.get("embeddings")
    if embeddings:
        return embeddings
    raise RuntimeError("Ollama did not return embeddings.")


class MixedRAGStore:
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        chromadb = require_chromadb()
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0

        ids = [stable_id(document) for document in documents]
        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]
        embeddings = ollama_embed(texts, model=self.embedding_model)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(documents)

    def ingest_pdf(self, path: str, chunk_size: int = 1200, chunk_overlap: int = 250) -> int:
        pages = load_pdf(path)
        chunks = split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return self.add_documents(chunks)

    def ingest_code(self, path: str, language: Optional[str] = None) -> int:
        chunks = tree_sitter_code_chunks(path, language=language)
        return self.add_documents(chunks)

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query cannot be empty")

        query_embedding = ollama_embed([query], model=self.embedding_model)[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            RetrievedChunk(
                text=text,
                metadata=metadata or {},
                distance=float(distance) if distance is not None else None,
            )
            for text, metadata, distance in zip(documents, metadatas, distances)
        ]


class Agent:
    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        max_steps: int = 8,
        rag_db_path: str = DEFAULT_DB_PATH,
        rag_collection: str = DEFAULT_COLLECTION,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.max_steps = max_steps
        self.rag_db_path = rag_db_path
        self.rag_collection = rag_collection
        self._rag_store: Optional[MixedRAGStore] = None
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt()}
        ]
        self.tools: List[Dict[str, Any]] = []
        self.setup_tools()

    @staticmethod
    def build_system_prompt() -> str:
        return (
            "You are Claw Coder, a coding, research, and local RAG assistant.\n\n"
            "Use search_knowledge_base for questions about ingested PDFs or local code.\n"
            "Use ingest_pdf_knowledge or ingest_code_knowledge only when the user asks to ingest files.\n"
            "Use search_stuff for outside web facts or current information.\n"
            "Use run_terminal only for local commands, tests, and file inspection.\n"
            "Use retrieved context directly. If context is insufficient, say what is missing.\n"
        )

    def rag_store(self) -> MixedRAGStore:
        if self._rag_store is None:
            self._rag_store = MixedRAGStore(
                db_path=self.rag_db_path,
                collection_name=self.rag_collection,
                embedding_model=self.embedding_model,
            )
        return self._rag_store

    def setup_tools(self) -> None:
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Search ingested PDFs and source code using ChromaDB RAG.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 4},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ingest_code_knowledge",
                    "description": "Ingest a local source-code file into the Tree-sitter RAG database.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "language": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ingest_pdf_knowledge",
                    "description": "Ingest a local PDF into the RAG database.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_stuff",
                    "description": "Search the internet for current or outside information.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "open_default_browser",
                    "description": "Open a URL in the default browser.",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_terminal",
                    "description": "Run a local terminal command.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout": {"type": "integer", "default": 30},
                        },
                        "required": ["command"],
                    },
                },
            },
        ]

    @staticmethod
    def parse_tool_arguments(raw_args: Any) -> Dict[str, Any]:
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                return {"value": raw_args}
        return {}

    @staticmethod
    def trim_text(value: str, limit: int = 700) -> str:
        return value if len(value) <= limit else value[:limit].rstrip() + "...(truncated)"

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        try:
            if tool_name == "search_knowledge_base":
                return self._search_knowledge_base_tool(tool_input)
            if tool_name == "ingest_code_knowledge":
                return self._ingest_code_tool(tool_input)
            if tool_name == "ingest_pdf_knowledge":
                return self._ingest_pdf_tool(tool_input)
            if tool_name == "search_stuff":
                return self._search_tool(tool_input)
            if tool_name == "open_default_browser":
                return self._open_browser_tool(tool_input)
            if tool_name == "run_terminal":
                return self._run_terminal_tool(tool_input)
            return json.dumps({"status": "error", "error": f"Unknown tool: {tool_name}"})
        except Exception as exc:
            logging.error("Tool failed: %s", exc)
            return json.dumps(
                {"status": "error", "tool": tool_name, "error": str(exc)},
                ensure_ascii=False,
            )

    def _search_knowledge_base_tool(self, tool_input: Dict[str, Any]) -> str:
        query = str(tool_input.get("query", "")).strip()
        top_k = int(tool_input.get("top_k", 4))
        if not query:
            return json.dumps({"status": "error", "error": "Missing query"})

        chunks = self.rag_store().search(query=query, top_k=top_k)
        return json.dumps(
            {
                "status": "ok",
                "query": query,
                "chunks": [
                    {
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "distance": chunk.distance,
                    }
                    for chunk in chunks
                ],
            },
            ensure_ascii=False,
        )

    def _ingest_code_tool(self, tool_input: Dict[str, Any]) -> str:
        path = str(tool_input.get("path", "")).strip()
        raw_language = tool_input.get("language")
        language = str(raw_language).strip() if raw_language is not None else None
        language = language or None
        if not path:
            return json.dumps({"status": "error", "error": "Missing path"})

        chunks_preview = tree_sitter_code_chunks(path, language=language)
        count = self.rag_store().add_documents(chunks_preview)
        return json.dumps(
            {"status": "ok", "path": path, "chunks_added": count},
            ensure_ascii=False,
        )

    def _ingest_pdf_tool(self, tool_input: Dict[str, Any]) -> str:
        path = str(tool_input.get("path", "")).strip()
        if not path:
            return json.dumps({"status": "error", "error": "Missing path"})

        count = self.rag_store().ingest_pdf(path)
        return json.dumps(
            {"status": "ok", "path": path, "chunks_added": count},
            ensure_ascii=False,
        )

    def search_info(self, query: str, max_results: int = 5) -> Optional[str]:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                if not results:
                    return "No search results"
                formatted = []
                for index, result in enumerate(results, start=1):
                    formatted.append(
                        f"Result {index}\n"
                        f"Title: {result.get('title', 'No title')}\n"
                        f"Snippet: {result.get('body', 'No body')}\n"
                        f"Source: {result.get('href', '')}"
                    )
                return "\n\n".join(formatted)
        except Exception as exc:
            logging.error("Search failed: %s", exc)
            return None

    def _search_tool(self, tool_input: Dict[str, Any]) -> str:
        query = str(tool_input.get("query", "")).strip()
        if not query:
            return json.dumps({"status": "error", "error": "Missing query"})

        result = self.search_info(query)
        if result is None:
            return json.dumps(
                {
                    "status": "error",
                    "query": query,
                    "error": "Search failed. Check internet/search provider.",
                }
            )
        return json.dumps({"status": "ok", "query": query, "results": result}, ensure_ascii=False)

    def _open_browser_tool(self, tool_input: Dict[str, Any]) -> str:
        url = str(tool_input.get("url", "")).strip()
        if not url:
            return json.dumps({"status": "error", "error": "Missing url"})
        if not urlparse(url).scheme:
            url = f"https://{url}"
        try:
            opened = bool(webbrowser.open(url, new=2))
        except Exception as exc:
            return json.dumps({"status": "error", "url": url, "error": str(exc)})
        if not opened:
            return json.dumps({"status": "error", "url": url, "error": "Browser could not open"})
        return json.dumps({"status": "ok", "url": url})

    @staticmethod
    def is_read_only_command(command: str) -> bool:
        try:
            parts = shlex.split(command)
        except ValueError:
            return False
        if not parts:
            return True
        if any(symbol in command for symbol in [">", ">>", "2>", "| tee", "&&", "||"]):
            return False
        if parts[0] in {"ls", "pwd", "whoami", "cat", "head", "tail", "grep", "find", "date", "echo", "wc"}:
            return True
        if parts[0] == "git" and len(parts) > 1:
            return parts[1] in {"status", "log", "show", "diff", "branch", "remote", "rev-parse"}
        if parts[0] in {"python", "python3"}:
            return any(flag in parts for flag in ("--version", "-V"))
        return False

    def needs_confirmation(self, command: str) -> bool:
        lowered = f" {command.strip().lower()} "
        high_risk = [
            "sudo ",
            " rm ",
            "rm -",
            "mv ",
            "cp ",
            "chmod ",
            "chown ",
            "git commit",
            "git push",
            "git reset",
            "git clean",
            "pip install",
            "pip uninstall",
        ]
        return any(marker in lowered for marker in high_risk) or not self.is_read_only_command(command)

    def ask_user_confirmation(self, command: str) -> bool:
        print("\nTool requested this terminal command:")
        print(f"  {command}")
        answer = input("Run this command? [y/N]: ").strip().lower()
        return answer in {"y", "yes"}

    @staticmethod
    def decode_process_output(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def run_terminal(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=max(1, timeout),
            )
            return {
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "command": command,
                "stdout": self.decode_process_output(exc.stdout),
                "stderr": f"Command timed out after {timeout} seconds.",
                "returncode": 124,
            }

    def _run_terminal_tool(self, tool_input: Dict[str, Any]) -> str:
        command = str(tool_input.get("command", "")).strip()
        if not command:
            return json.dumps({"status": "error", "error": "Missing command"})
        timeout = int(tool_input.get("timeout", 30))
        if self.needs_confirmation(command) and not self.ask_user_confirmation(command):
            return json.dumps({"status": "cancelled", "command": command})

        result = self.run_terminal(command, timeout=timeout)
        status = "ok" if result["returncode"] == 0 else "error"
        return json.dumps({"status": status, "result": result}, ensure_ascii=False)

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        for _ in range(self.max_steps):
            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                stream=False,
            )
            message = response.get("message", {})
            assistant_message = {"role": "assistant", "content": message.get("content", "")}
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            self.messages.append(assistant_message)

            if not tool_calls:
                return message.get("content", "")

            for call in tool_calls:
                function_data = call.get("function", {})
                tool_name = function_data.get("name", "")
                tool_args = self.parse_tool_arguments(function_data.get("arguments", {}))
                result = self.execute_tool(tool_name, tool_args)
                self.messages.append({"role": "tool", "content": result})

        return "I reached the tool-execution step limit before finishing."


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def preview_code_chunks(path: str, language: Optional[str] = None) -> None:
    chunks = tree_sitter_code_chunks(path, language=language)
    print_json(
        [
            {
                "source": chunk.metadata.get("source"),
                "language": chunk.metadata.get("language"),
                "symbol_type": chunk.metadata.get("symbol_type"),
                "symbol_name": chunk.metadata.get("symbol_name"),
                "start_point": chunk.metadata.get("start_point"),
                "end_point": chunk.metadata.get("end_point"),
                "text": Agent.trim_text(chunk.page_content, 500),
            }
            for chunk in chunks
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone RAG agent")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("chat")
    subparsers.add_parser("languages")

    code_chunks = subparsers.add_parser("code-chunks")
    code_chunks.add_argument("path")
    code_chunks.add_argument("--language")

    ingest_code = subparsers.add_parser("ingest-code")
    ingest_code.add_argument("path")
    ingest_code.add_argument("--language")

    ingest_pdf = subparsers.add_parser("ingest-pdf")
    ingest_pdf.add_argument("path", nargs="?", default=str(DEFAULT_PDF))

    search_kb = subparsers.add_parser("search-kb")
    search_kb.add_argument("query")
    search_kb.add_argument("--top-k", type=int, default=4)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "languages":
        print_json(available_languages())
        return

    if args.command == "code-chunks":
        preview_code_chunks(args.path, language=args.language)
        return

    agent = Agent(
        model=args.model,
        embedding_model=args.embedding_model,
        rag_db_path=args.db_path,
        rag_collection=args.collection,
    )

    if args.command == "ingest-code":
        print(agent.execute_tool("ingest_code_knowledge", {"path": args.path, "language": args.language}))
        return
    if args.command == "ingest-pdf":
        print(agent.execute_tool("ingest_pdf_knowledge", {"path": args.path}))
        return
    if args.command == "search-kb":
        print(agent.execute_tool("search_knowledge_base", {"query": args.query, "top_k": args.top_k}))
        return
    if args.command == "chat":
        print("Claw Coder RAG. Type 'exit' to quit.\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() in {"exit", "quit"}:
                break
            print(f"\nAgent: {agent.chat(user_input)}\n")


if __name__ == "__main__":
    main()

