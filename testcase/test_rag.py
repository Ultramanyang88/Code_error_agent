import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from rag.indexer import RepoIndexer
from rag.retrieve import RAGEngine


def main():
    repo_root = str(ROOT_DIR)

    print("===== Building FAISS index =====")
    indexer = RepoIndexer(repo_root=repo_root)
    index, chunks = indexer.build(force_rebuild=True)
    print(f"Built index with {len(chunks)} chunks.")

    print("\n===== Retrieve Query 1 =====")
    engine = RAGEngine(repo_root=repo_root, auto_load=True)
    results = engine.retrieve(
        query="executor tool call run step state memory",
        top_k=5,
    )
    print(engine.format_context(results))

    print("\n===== Retrieve Query 2 =====")
    results = engine.retrieve(
        query="planner creates initial plan and adjusts plan after failure",
        top_k=5,
    )
    print(engine.format_context(results))

    print("\n===== Retrieve Query 3 =====")
    results = engine.retrieve(
        query="read file search code run tests apply patch tools",
        top_k=5,
    )
    print(engine.format_context(results))


if __name__ == "__main__":
    main()