"""ClaudeYe: a vision middleware (MCP server) that gives Claude Code precise eyes for images and videos."""

__version__ = "0.1.0"


def main() -> None:
    from .server import run

    run()
