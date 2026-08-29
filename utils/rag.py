"""
STEP 2: RAG (Retrieval Augmented Generation)
----------------------------------------------
The idea is simple: a job description can be long (2-3 pages).
If we send the whole JD to the AI every time, that:
  a) costs more (more tokens)
  b) can confuse the AI with irrelevant details

So we split the JD into small "chunks", generate embeddings
(a numeric representation of meaning) for each, and when we
need to analyze a resume, we only retrieve the most RELEVANT
chunks.

We use ChromaDB (a local vector database) here - no cloud setup
required, everything runs on your own machine.
"""

import chromadb
from chromadb.utils import embedding_functions

# Free, local embedding model - downloads once from the internet,
# then runs without needing an API key
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits text into smaller pieces.
    chunk_size = how many words per chunk
    overlap = how much overlap between chunks (keeps context from breaking)
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # keep a bit of overlap
    return chunks


class JDRetriever:
    """
    Class for storing and searching a job description.
    We create a fresh collection for every new JD.
    """

    def __init__(self, collection_name="jd_collection"):
        # in-memory client - data resets on restart (fine for this project)
        self.client = chromadb.Client()
        # delete the collection if it already exists, to start fresh
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=collection_name, embedding_function=embedding_fn
        )

    def index_jd(self, jd_text):
        """Split the JD into chunks and store them in the database"""
        chunks = chunk_text(jd_text)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, ids=ids)
        return len(chunks)

    def retrieve_relevant_chunks(self, query, top_k=4):
        """
        Find the most relevant JD chunks against a query
        (e.g. the resume's skills section).
        """
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results["documents"][0]  # list of relevant chunk texts