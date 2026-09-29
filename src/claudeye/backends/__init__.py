"""Optional model backends. Each is imported lazily by models_tools; a missing dependency
surfaces as an install hint instead of breaking the server."""

INSTALL_HINTS = {
    "ocr": "uv sync --extra ocr",
    "asr": "uv sync --extra asr   (GPU: also --extra asr-cuda)",
    "ml": "uv sync --extra ml --extra cpu   (RTX PC: --extra ml --extra cu130)",
    "gemini": "uv sync --extra gemini  and set GEMINI_API_KEY",
    "ollama": "install Ollama (https://ollama.com), then: ollama pull qwen3.5:9b",
}


class MissingBackend(ImportError):
    def __init__(self, what: str, key: str):
        super().__init__(f"{what} is not installed. Install with: {INSTALL_HINTS[key]}")
