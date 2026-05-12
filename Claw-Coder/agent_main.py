from __future__ import annotations

import ollama
import importlib
import json
import logging
import argparse
import subprocess
import hashlib
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional
from urllib.parse import urlparse
from ddgs import DDGS
import shlex


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("agent_main.log"), logging.StreamHandler(sys.stdout)],

)

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_CHAT_MODEL="llama3.2:3b"
DEFAULT_EMBEDDING_MODEL="qwen3-embedding-4b"
DEFAULT_DB_PATH = "agent_main_chroma_db"
DEFAULT_COLLECTION="agent_knowledge"
DEFAULT_PDF = Path(__file__).resolve().parent/"data"/"2509.24435v1.pdf"

LANGUAGES_SPECS = Dict[str, Dict[str, str]] = {
    "python": {"module": "tree_sitter_python", "function": "language"},
    "javascript": {"module": "tree_sitter_javascript", "function": "language"},
    "typescript": {"module": "tree_sitter_typescript", "function": "language_typescript"},
    "tsx": {"module": "tree_sitter_typescript", "function": "language_tsx"},
    "json": {"module": "tree_sitter_json", "function": "language"},
    "html": {"module": "tree_sitter_html", "function": "language"},
    "c#": {"module": "tree_sitter_c#", "function": "language"},
    "c++": {"module": "tree_sitter_c++", "function": "language"},
    "c": {"module": "tree_sitter_c", "function": "language"},
    "go": {"module": "tree_sitter_go", "function": "language"},
    "java": {"module": "tree_sitter_java", "function": "language"},
    "rust": {"module": "tree_sitter_rust", "function": "language"},

}
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".rs": "rust",
    ".cjs": "javascript",
    "js": "javascript",
    "mjs": "javascript",
    ".c#": "c#",
    ".java": "java",
    ".c": "c",
    ".c++": "c++",
    ".html": "html",
    ".css": "css",
    ".ts": "typescript",
    ".json": "json",
    "tsx": "tsx",
}

@dataclass(slots=True)
class Document:
    page_content: str
    meta_data: Dict[str, Any]
@dataclass(slots=True)
class RetrievedChunk:
    text: str
    meta_data: Dict[str, Any]
    distance: Optional[float]

def require_chromadb():
    try:
        import chromadb
    except TimeoutError as exc:
        raise RuntimeError("Chromadb is missing in your environment, You can import it using pip install chromadb") from exc
    return chromadb
def require_pdf_reader():
    try:
        from pypdf import PdfReader
    except ImportError as excs:
        raise RuntimeError("pypdf is missing in your environment, You should probably import it using pip install pypdf") from excs
    return PdfReader
def require_tree_sitter():
    try:
        from tree_sitter import Language, Parser, Query, QueryCursor
    except ImportError as e:
        raise RuntimeError("tree_sitter is missing in your environment, You should install that too using pip install tree_sitter") from e
    return Language, Parser, Query, QueryCursor

def available_languages() -> Dict[str, Dict[str, Any]]:
    status: Dict[str, Dict[str, Any]] = {}
    for language,  spec in LANGUAGES_SPECS.items():
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
def load_tree_sitter_language(language_name: str) :
    Language, Parser, Query, QueryCursor = require_tree_sitter()
    spec = LANGUAGES_SPECS.get(language_name)
    if not spec:
        supported = ", ".join(sorted(LANGUAGES_SPECS))
        raise RuntimeError(f"This is not a supported language {language_name}. Supported languages: {supported}")

    try:
        module = importlib.import_module(spec["module"])
        language_fn = getattr(module, spec["function"])
    except Exception as e:
        package = spec["module"].replace("_", "-")
        raise RuntimeError(
            f"Tree-sitter grammar for {language_name} missing you can install it with: "
            f"pip install {package}"
        )from e

    parser = Parser()
    parser.language = Language(language_fn)
    return parser, Query, QueryCursor, parser.language

def node_text(source: bytes, node: Any) -> str:






