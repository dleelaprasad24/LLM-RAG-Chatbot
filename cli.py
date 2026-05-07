"""Command-line interface for ingestion and interactive chat.

Examples
--------
    # Ingest a directory of PDFs:
    python cli.py ingest --path ./data

    # Ingest a single file:
    python cli.py ingest --path ./data/handbook.pdf

    # Ingest from URLs:
    python cli.py ingest --url https://example.com/article \\
                         --url https://example.com/post

    # Start an interactive chat session:
    python cli.py chat

    # Inspect or reset the vector store:
    python cli.py status
    python cli.py reset
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.ingest import ingest_path, ingest_urls
from src.rag_chain import RAGChatbot, build_chat_history
from src.vector_store import count_documents, reset_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-cli")


def cmd_ingest(args: argparse.Namespace) -> int:
    total = 0
    if args.path:
        for p in args.path:
            total += ingest_path(p)
    if args.url:
        total += ingest_urls(args.url)
    if not args.path and not args.url:
        print("Provide --path and/or --url. See `python cli.py ingest -h`.")
        return 1
    print(f"Done. Added {total} chunks. Total vectors: {count_documents()}")
    return 0


def cmd_chat(_: argparse.Namespace) -> int:
    if count_documents() == 0:
        print("No documents have been ingested. Run `python cli.py ingest --path ./data` first.")
        return 1

    print("RAG Chatbot ready. Type 'exit' or 'quit' to leave.\n")
    bot = RAGChatbot()
    history: list[tuple[str, str]] = []

    try:
        while True:
            try:
                question = input("You: ").strip()
            except EOFError:
                print()
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit", ":q"}:
                break

            response = bot.ask(question, chat_history=build_chat_history(history))
            print(f"\nAssistant: {response.answer}\n")

            if response.sources:
                print("Sources:")
                for i, src in enumerate(response.formatted_sources(), start=1):
                    label = src.get("title") or src.get("source")
                    page = f" (page {src['page'] + 1})" if isinstance(src.get("page"), int) else ""
                    print(f"  [{i}] {label}{page}  -  {src['source']}")
                print()

            history.append((question, response.answer))
    except KeyboardInterrupt:
        print("\nGoodbye!")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    print(f"Vectors stored: {count_documents()}")
    return 0


def cmd_reset(_: argparse.Namespace) -> int:
    reset_collection()
    print("Vector store cleared.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-cli", description="LLM-Powered RAG Chatbot CLI."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest files, folders, or URLs.")
    p_ingest.add_argument("--path", action="append", help="File or directory path.")
    p_ingest.add_argument("--url", action="append", help="URL to fetch and ingest.")
    p_ingest.set_defaults(func=cmd_ingest)

    p_chat = sub.add_parser("chat", help="Start an interactive chat session.")
    p_chat.set_defaults(func=cmd_chat)

    p_status = sub.add_parser("status", help="Show vector store stats.")
    p_status.set_defaults(func=cmd_status)

    p_reset = sub.add_parser("reset", help="Delete every vector in the collection.")
    p_reset.set_defaults(func=cmd_reset)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
