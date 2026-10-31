
import json
import re
from pathlib import Path

import mock_llm
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

POLICY_DIR = Path("data/policies")
RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 40
OVERLAP = 10
STEP_SIZE = CHUNK_SIZE - OVERLAP

FALLBACK_ANSWER = (
    "I could not find that information in the policy documents."
)


# ============================================================
# PART 1: DOCUMENT CHUNKING
# ============================================================

def chunk_document(text, source):
    """Split a document into overlapping word chunks."""

    words = text.split()
    chunks = []

    for start in range(0, len(words), STEP_SIZE):
        chunk_words = words[start:start + CHUNK_SIZE]

        if not chunk_words:
            break

        chunks.append({
            "source": source,
            "chunk_id": len(chunks),
            "start_word": start,
            "text": " ".join(chunk_words),
        })

        if start + CHUNK_SIZE >= len(words):
            break

    return chunks


def main():
    """Create policy chunks and verify reference counts."""

    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy folder not found: {POLICY_DIR}"
        )

    policy_files = sorted(POLICY_DIR.glob("*.txt"))

    if len(policy_files) != 5:
        raise ValueError(
            f"Expected 5 policy files, found {len(policy_files)}"
        )

    all_chunks = []

    print("\nPOLICY CHUNKING CHECKPOINTS")

    for path in policy_files:
        text = path.read_text(
            encoding="utf-8"
        ).strip()

        document_chunks = chunk_document(
            text,
            path.name
        )

        all_chunks.extend(document_chunks)

        print(
            f"{path.name}: "
            f"{len(document_chunks)} chunks"
        )

    output_path = OUTPUT_DIR / "policy_chunks.json"

    output_path.write_text(
        json.dumps(
            all_chunks,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nTotal chunks:", len(all_chunks))
    print("Saved:", output_path)

    expected = {
        "refund_policy.txt": 4,
        "delivery_policy.txt": 3,
        "escalation_policy.txt": 3,
        "payment_policy.txt": 3,
        "privacy_policy.txt": 2,
    }

    actual = {
        path.name: sum(
            chunk["source"] == path.name
            for chunk in all_chunks
        )
        for path in policy_files
    }

    if actual != expected or len(all_chunks) != 15:
        raise AssertionError(
            f"Chunk counts differ from reference: {actual}"
        )

    print("PASS: all 15 policy chunks match the reference.")


# ============================================================
# PART 2: LOAD CHUNKS
# ============================================================

def load_chunks():
    """Load the generated JSON chunks."""

    path = OUTPUT_DIR / "policy_chunks.json"

    if not path.exists():
        raise FileNotFoundError(
            "policy_chunks.json not found. "
            "Run the chunking step first."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# PART 3: TF-IDF RETRIEVAL
# ============================================================

def retrieve(query, k=3, chunks=None):
    """Retrieve the top-k chunks using cosine similarity."""

    if chunks is None:
        chunks = load_chunks()

    if not chunks or not query.strip():
        return []

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    if not any(text.strip() for text in texts):
        return []

    vectorizer = TfidfVectorizer()

    document_vectors = vectorizer.fit_transform(texts)

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: scores[index],
        reverse=True
    )

    return [
        {
            "source": chunks[index]["source"],
            "score": round(
                float(scores[index]), 4
            ),
            "text": chunks[index]["text"],
        }
        for index in ranked_indices[:max(0, k)]
    ]


# ============================================================
# PART 4: READ COMPLETE RETRIEVED DOCUMENTS
# ============================================================

def load_source_documents(source_names):
    """
    Read complete text from retrieved source documents.

    This helps preserve important facts that might otherwise
    be split across chunk boundaries.
    """

    documents = []
    seen = set()

    for source in source_names:

        if source in seen:
            continue

        seen.add(source)

        path = POLICY_DIR / source

        if not path.exists():
            continue

        text = path.read_text(
            encoding="utf-8"
        ).strip()

        if text:
            documents.append((source, text))

    return documents


# ============================================================
# PART 5: RAG ANSWER GENERATION
# ============================================================

def answer(question, chunks=None):
    """Generate an answer using retrieved policy context."""

    results = retrieve(
        question,
        k=3,
        chunks=chunks
    )

    if not results:
        return FALLBACK_ANSWER, []

    # Reject queries with no meaningful lexical match.
    if results[0]["score"] <= 0:
        return FALLBACK_ANSWER, []

    source_names = list(dict.fromkeys(
        result["source"]
        for result in results
    ))

    documents = load_source_documents(source_names)

    if not documents:
        return FALLBACK_ANSWER, []

    context = "\n\n".join(
        f"[{source}]\n{text}"
        for source, text in documents
    )

    system_prompt = """
You are a POLICY ASSISTANT.

Answer the question using ONLY the supplied policy context.

Rules:
1. Give a direct and complete answer.
2. Preserve exact numbers, amounts and time limits.
3. Include the important policy condition.
4. Do not invent facts or use outside knowledge.
5. If the answer is not available, respond exactly:
I could not find that information in the policy documents.
"""

    user_prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )

    response = mock_llm.chat([
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ])

    answer_text = (
        response.choices[0].message.content or ""
    ).strip()

    if not answer_text:
        return FALLBACK_ANSWER, []

    sources = [
        source
        for source, _ in documents
    ]

    return answer_text, sources


# ============================================================
# PART 6: RETRIEVAL TEST
# ============================================================

def test_retrieval():
    """Test retrieval against the reference query."""

    query = "refund policy late delivery"

    results = retrieve(query, k=3)

    print("\nRETRIEVAL TEST")
    print("Query:", query)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank {rank}")
        print("Source:", result["source"])
        print("Score:", result["score"])
        print("Text:", result["text"])

    assert len(results) == 3, (
        "Expected 3 retrieved chunks"
    )

    assert all(
        result["source"] == "refund_policy.txt"
        for result in results
    ), "Top three chunks do not match the reference"

    print("\nPASS: retrieval matches the reference.")


# ============================================================
# PART 7: ANSWER TEST
# ============================================================

def test_answer():
    """Test basic RAG answer generation."""

    question = (
        "What delivery time makes a completed order "
        "eligible for a full refund?"
    )

    answer_text, sources = answer(question)

    print("\nRAG ANSWER TEST")
    print("Question:", question)
    print("Answer:", answer_text)
    print("Sources:", sources)

    assert answer_text, "Answer must not be empty."

    assert sources, "At least one source must be returned."

    assert "refund_policy.txt" in sources, (
        "Refund policy source was not returned."
    )

    print("PASS: RAG answer generation works.")


# ============================================================
# PART 8: TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """Normalize punctuation and spaces for comparisons."""

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(text).lower()
    ).strip()


# ============================================================
# PART 9: ZERO-OVERLAP CHUNKING
# ============================================================

def make_zero_overlap_chunks():
    """Create 40-word chunks without overlap."""

    chunks = []

    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy folder not found: {POLICY_DIR}"
        )

    for path in sorted(POLICY_DIR.glob("*.txt")):

        text = path.read_text(
            encoding="utf-8"
        ).strip()

        words = text.split()
        chunk_id = 0

        for start in range(0, len(words), CHUNK_SIZE):

            chunk_words = words[
                start:start + CHUNK_SIZE
            ]

            if not chunk_words:
                continue

            chunks.append({
                "source": path.name,
                "chunk_id": chunk_id,
                "start_word": start,
                "text": " ".join(chunk_words),
            })

            chunk_id += 1

    return chunks


# ============================================================
# PART 10: POLICY RAG EVALUATION
# ============================================================

def evaluate_policy_rag(chunks, label):
    """
    Evaluate:
    - Expected document retrieval
    - Expected fact in generated answers
    - Fallback for unanswerable questions
    """

    eval_path = RAW_DIR / "policy_eval.csv"

    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {eval_path}"
        )

    df = pd.read_csv(eval_path).fillna("")

    required_columns = {
        "question",
        "expected_doc",
        "expected_fact"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing evaluation columns: {missing_columns}"
        )

    answerable = 0
    retrieval_hits = 0
    fact_hits = 0
    fallback_hits = 0
    fallback_total = 0

    failures = []

    print(f"\n{'=' * 55}")
    print("EVALUATION:", label)
    print("=" * 55)

    for _, row in df.iterrows():

        question = str(row["question"]).strip()
        expected_doc = str(row["expected_doc"]).strip()
        expected_fact = str(row["expected_fact"]).strip()

        results = retrieve(
            question,
            k=3,
            chunks=chunks
        )

        retrieved_docs = [
            result["source"]
            for result in results
        ]

        # Unanswerable questions have no expected document.
        if (
            not expected_doc
            or expected_doc.upper() == "NONE"
        ):

            fallback_total += 1

            answer_text, _ = answer(
                question,
                chunks=chunks
            )

            fallback = (
                "could not find that information"
                in normalize_text(answer_text)
            )

            fallback_hits += int(fallback)

            print(f"\nUNANSWERABLE: {question}")
            print("Fallback correct:", fallback)
            print("Answer:", answer_text)

            if not fallback:
                failures.append((
                    question,
                    "Expected fallback answer"
                ))

            continue

        answerable += 1

        doc_hit = expected_doc in retrieved_docs

        retrieval_hits += int(doc_hit)

        answer_text, sources = answer(
            question,
            chunks=chunks
        )

        normalized_answer = normalize_text(answer_text)
        normalized_fact = normalize_text(expected_fact)

        fact_hit = (
            bool(normalized_fact)
            and normalized_fact in normalized_answer
        )

        fact_hits += int(fact_hit)

        print(f"\nQuestion: {question}")
        print(f"Expected document: {expected_doc}")
        print(f"Retrieved documents: {retrieved_docs}")
        print(f"Sources used: {sources}")
        print(f"Retrieval hit: {doc_hit}")
        print(f"Expected fact: {expected_fact}")
        print(f"Answer: {answer_text}")
        print(f"Fact hit: {fact_hit}")

        if not doc_hit:
            failures.append((
                question,
                "Expected document not retrieved"
            ))

        if not fact_hit:
            failures.append((
                question,
                "Expected fact not found in answer"
            ))

    print(f"\n--- {label} SUMMARY ---")
    print(
        f"Answerable questions: {answerable}"
    )
    print(
        f"Expected document retrieved: "
        f"{retrieval_hits}/{answerable}"
    )
    print(
        f"Expected fact matched: "
        f"{fact_hits}/{answerable}"
    )
    print(
        f"Unanswerable fallback correct: "
        f"{fallback_hits}/{fallback_total}"
    )

    print("\nFAILURES:")

    if failures:
        for question, reason in failures:
            print(f"- {question}: {reason}")
    else:
        print("No evaluation failures.")

    return failures


# ============================================================
# PART 11: RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    main()

    test_retrieval()

    test_answer()

    # Evaluate original chunks with 10-word overlap.
    original_chunks = load_chunks()

    evaluate_policy_rag(
        original_chunks,
        "WITH 10-WORD OVERLAP"
    )

    # Evaluate 40-word chunks without overlap.
    zero_overlap_chunks = make_zero_overlap_chunks()

    evaluate_policy_rag(
        zero_overlap_chunks,
        "ZERO OVERLAP"
    )

    print("\nPART D RAG PIPELINE COMPLETED")
