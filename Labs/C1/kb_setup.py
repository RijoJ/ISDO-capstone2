"""
KB ingestion + retrieval test for ISDO (Intelligent Service Desk Orchestrator).

What it does:
1. Reads all .md files from a KB folder.
2. Splits each file into chunks at '## ' headings (each chunk keeps its
   heading as the first line, and any content before the first heading
   is kept as an intro chunk).
3. Stores every chunk in a ChromaDB collection called 'isdo_kb', with
   metadata pointing back to the source article and heading.
4. Runs 4 sample queries and prints, for each, the best-matching
   ARTICLE (not just chunk) and a confidence score.

Dependencies: chromadb only (uses its default built-in embedding
function, so no extra embedding package is required).
"""

import glob
import os
import re

import chromadb

KB_DIR = "data/kb"
COLLECTION_NAME = "isdo_kb"
CHROMA_PATH = "chroma_db"  # persisted on disk so re-runs don't re-embed

# Sample queries an L1 agent / requester might type at ticket intake.
SAMPLE_QUERIES = [
    "my VPN keeps disconnecting every few minutes",
    "I can't reset my password on the portal",
    "employee laptop is completely frozen and won't turn on",
    "getting an error when I try to log in to the company app",
]


def split_into_chunks(text: str, source_file: str):
    """
    Split markdown text into chunks at '## ' headings.

    Returns a list of dicts: {"text": chunk_text, "heading": heading_title}
    Content before the first '## ' heading (e.g. a top-level '# Title')
    is kept as its own chunk with heading "Intro".
    """
    # Split but keep the heading lines using a capturing group.
    parts = re.split(r"(?m)^(##\s+.*)$", text)

    chunks = []

    # parts[0] is whatever came before the first '## ' heading.
    intro = parts[0].strip()
    if intro:
        chunks.append({"text": intro, "heading": "Intro"})

    # Remaining parts alternate: heading, body, heading, body, ...
    for i in range(1, len(parts), 2):
        heading_line = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        heading_title = heading_line.lstrip("#").strip()
        chunk_text = f"{heading_line}\n{body}".strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "heading": heading_title})

    return chunks


def load_and_chunk_kb(kb_dir: str):
    """Read every .md file in kb_dir and return flat lists ready for Chroma."""
    md_files = sorted(glob.glob(os.path.join(kb_dir, "*.md")))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in '{kb_dir}'")

    documents, metadatas, ids = [], [], []

    for file_path in md_files:
        article_name = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = split_into_chunks(text, file_path)

        for idx, chunk in enumerate(chunks):
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "article": article_name,
                    "source_file": file_path,
                    "heading": chunk["heading"],
                    "chunk_index": idx,
                }
            )
            ids.append(f"{article_name}::chunk_{idx}")

    return documents, metadatas, ids


def build_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Start clean each run so re-ingesting doesn't create duplicate chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    documents, metadatas, ids = load_and_chunk_kb(KB_DIR)
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(f"Ingested {len(documents)} chunks from {len(set(m['article'] for m in metadatas))} articles "
          f"into collection '{COLLECTION_NAME}'.\n")

    return collection


def best_article_for_query(collection, query: str, n_results: int = 5):
    """
    Query the collection and collapse chunk-level hits down to the
    single best-matching ARTICLE, using each chunk's distance to derive
    a 0-1 confidence score (1 = perfect match, closer to 0 = weak match).
    """
    results = collection.query(query_texts=[query], n_results=n_results)

    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    if not distances:
        return None, 0.0

    # Chroma's default embedding function uses cosine distance in [0, 2].
    # Convert to an intuitive 0-1 confidence: confidence = 1 - distance/2.
    best_distance = distances[0]
    best_article = metadatas[0]["article"]
    confidence = max(0.0, 1.0 - (best_distance / 2.0))

    return best_article, confidence


def main():
    collection = build_collection()

    print("Sample query results:\n")
    for query in SAMPLE_QUERIES:
        article, confidence = best_article_for_query(collection, query)
        print(f"Query: {query}")
        if article is None:
            print("  No match found.\n")
            continue
        print(f"  Best matching article: {article}")
        print(f"  Confidence score:      {confidence:.4f}\n")


if __name__ == "__main__":
    main()