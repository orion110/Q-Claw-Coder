QClawCoder- merged Q-Claw + Claw-Coder.

🐾 QClawCode (Q-Claw + Claw-Coder)
Autonomous local AI agent for your terminal. Merges a conversational REPL persona with tool-calling RAG, voice interaction, and safe terminal execution. 100% Local, 100% Private.

QClawCode is a self-contained, privacy-first AI powerhouse that bridges the gap between a terminal chatbot and a autonomous developer agent. Built by merging the voice-enabled terminal persona Q-Claw and the pragmatic, tool-calling RAG specialist Claw-Coder, it can parse code, ingest documents, search the web, and execute shell commands—all via local models.

✨ Key Features
🧠 Tool-Calling Agent: Hooked directly into the Ollama Tools API. The agent autonomously decides whether to search your local knowledge base, ingest a new file, search the web, or execute terminal commands to answer your question. (Includes a graceful fallback for models that don't support tools).
📚 Multi-Language Code RAG: Uses Tree-sitter to parse source code across 13 languages (Python, Rust, Go, C/C++/C#, JS/TS, etc.), extracting specific functions and classes rather than just blindly chunking text.
🕸️ Knowledge Graphing: As it ingests code, it builds a local JSON knowledge graph mapping out files and symbols. It uses this graph to "rerank" vector search results for highly accurate retrieval.
🔒 Safe Terminal Execution: If the agent needs to run a command, it will. But if the command is destructive (like rm, sudo, or git push), it pauses and asks for explicit user confirmation first.
🎙️ Voice & Multimodal: Born from Q-Claw, it possesses a voice. Using Kokoro TTS and Vosk STT, it can lock into a continuous "Voice Mode" where you speak to it, and it speaks back.
⚡ Offline Fast-Paths: Knows when not to bother the AI. If you say "hi", it responds instantly. If you ask for a quick web search, it bypasses the LLM entirely and fetches summaries from Wikipedia and DuckDuckGo.
🛠️ Tech Stack
Brain: Ollama (Local LLMs like llama3.2:3b or qwen2.5-coder:7b)
Memory: ChromaDB (Vector storage) + JSON Knowledge Graph
Parser: Tree-sitter (AST code parsing)
Senses: sounddevice, vosk, kokoro-onnx (STT/TTS)
Environment: Python 3.10+
🚀 Installation & Setup
1. Prerequisites
Ensure you have Ollama installed and running on your machine.Pull a model that supports tools (recommended):

ollama pull llama3.2:3bollama pull qwen3-embedding:4b  # Used for vector embeddings
2. Install QClawCode
Clone the repository and run the built-in setup command to install Python dependencies:

bash
￼
git clone https://github.com/gabriel-c70/Claw-Coder.git
cd Claw-Coder
python Qclawcode.py setup
3. Verify Setup
Run the doctor command to ensure your Node, Python, and Ollama environments are configured correctly:

bash
￼
python Qclawcode.py doctor
💻 Usage
Start the interactive chat REPL (default):

bash
￼
python Qclawcode.py
# or
python Qclawcode.py chat
CLI Commands
text
￼
╔══════════════════════════════════════════════════════════════════════════════╗
║                          CLAW CODER - Autonomous local AI agent              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📖 USAGE:
  claw <command> [options]

╔══════════════════════════════════════════════════════════════════════════════╗
║ 💬 CHAT & INTERACTION                                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  chat [--pdf <file>...]         Start interactive chat (optionally preload   ║
║                                PDFs)                                         ║
║  chat --ui textual              Use improved UI with scrolling & selection   ║
║  models                         List local Ollama models                     ║
║  <model-name>                   Start chat with specific Ollama model        ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║ 📚 KNOWLEDGE BASE                                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ingest <paths...>              Ingest files/directories into graph + vector ║
║                                RAG                                           ║
║  ingest-code <file>             Ingest one source file                       ║
║  ingest-pdf <file>              Ingest a PDF or text document (.pdf, .txt,   ║
║                                .md)                                          ║
║  search <query>                 Search vector RAG with graph reranking       ║
║  graph <query>                  Search the knowledge graph only              ║
║  summary                        Show graph node/edge counts                  ║
║  languages                      Show Tree-sitter language support            ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║ ⚙  SETUP & CONFIGURATION                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  setup                        Install Python dependencies for Claw Coder     ║
║  doctor                       Check local Node/Python/Ollama setup           ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║ 🔧 COMMON OPTIONS                                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  --top-k <n>                    Number of results to return                  ║
║  --depth <n>                    Graph traversal depth for graph search       ║
║  --graph <file>                 Knowledge graph JSON path                    ║
║  --db <dir>                     ChromaDB directory                           ║
║  --collection <name>            ChromaDB collection                          ║
║  --model <name>                 Ollama chat model                            ║
║  --embedding-model <name>       Ollama embedding model                       ║
║  --ui <rich|textual>            Choose UI style (default: rich)              ║
╚══════════════════════════════════════════════════════════════════════════════╝
REPL Slash Commands
Once inside the chat (python Qclawcode.py chat), you can use the following commands:

COMMAND
DESCRIPTION
/help	Show the REPL commands menu.
/search <query>	Wiki + web quick search (bypasses LLM).
/kb <query>	Search the local RAG knowledge base.
/ingest <file>	Ingest a file (pdf or code) into the knowledge base.
/model <name>	Switch the Ollama model on the fly.
/languages	Show Tree-sitter code-RAG language support.
/mic	Toggle mic on/off (Vosk STT).
/listen	Lock into continuous voice mode (speaks replies).
/reset	Wipe conversation history (fresh context).
/clear	Clear the terminal screen.
/exit	Quit QClawCode.
￼
🌳 Supported Code Languages
QClawCode uses Tree-sitter to intelligently parse the following languages:

Python (.py)
JavaScript (.js, .mjs, .cjs)
TypeScript (.ts, .tsx)
C / C++ (.c, .cpp, .cc, .h, .hpp)
C# (.cs)
Java (.java)
Go (.go)
Rust (.rs)
HTML, CSS, JSON



[![IMG-2978.webp](https://i.postimg.cc/90CktdGg/IMG-2978.webp)](https://postimg.cc/1n7MyN7D)
