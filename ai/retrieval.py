"""Retrieval: ai.embed, ai.cosine, ai.chunk and ai.Index."""

import math
import zlib
from typing import NamedTuple

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore

from ._engine import complete
from ._words import content_words

DIMENSIONS = 512


def embed(text):
    """Turn text into 512 numbers: a hashed bag of words, function words dropped.

    Honest enough to teach the geometry, but real embedding models are neural
    networks and understand meaning, not just shared words.
    """
    vector = [0.0] * DIMENSIONS
    for word in content_words(text):
        vector[zlib.crc32(word.encode()) % DIMENSIONS] += 1.0
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector] if length else vector


def cosine(a, b):
    """How alike two vectors point: 1 for the same direction, about 0 for unrelated.

    You may also pass two strings; they are embedded first.
    """
    a = embed(a) if isinstance(a, str) else a
    b = embed(b) if isinstance(b, str) else b
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def chunk(text):
    """Split a document into paragraphs at blank lines."""
    paragraphs, current = [], []
    for line in str(text).splitlines():
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


class HashEmbeddings(Embeddings):
    """ai.embed wrapped as a LangChain Embeddings object."""

    def embed_documents(self, texts):
        return [embed(t) for t in texts]

    def embed_query(self, text):
        return embed(text)


class Hit(NamedTuple):
    """One search result: how well it scored, the chunk's text, and its number."""

    score: float
    text: str
    index: int


class Index:
    """A searchable collection of chunks (a card catalogue).

        index = ai.Index(ai.chunk(handbook))
        index.search("When is the library open?")
        index.ask("When is the library open?")
    """

    def __init__(self, chunks=None):
        self.chunks = []
        self._store = InMemoryVectorStore(HashEmbeddings())
        for text in chunks or []:
            self.add(text)

    def add(self, text):
        """Add one chunk. Returns its number, which citations use."""
        number = len(self.chunks)
        self.chunks.append(str(text))
        self._store.add_texts([str(text)], ids=[str(number)], metadatas=[{"index": number}])
        return number

    def add_file(self, path):
        """Add every paragraph of a text file. Returns the new chunk numbers."""
        with open(path, encoding="utf-8") as f:
            return [self.add(paragraph) for paragraph in chunk(f.read())]

    def search(self, q, k=3):
        """The k best chunks for a question, best first, as Hit(score, text, index)."""
        if not self.chunks:
            return []
        found = self._store.similarity_search_with_score(str(q), k=k)
        return [Hit(round(score, 4), doc.page_content, doc.metadata["index"]) for doc, score in found]

    def best(self, q):
        """The single best Hit, or None if the index is empty."""
        hits = self.search(q, k=1)
        return hits[0] if hits else None

    def ask(self, q, k=3, threshold=0.15, refuse="I don't have that in my documents.", system=""):
        """Retrieve, augment and generate: answer from the best chunks, with a citation.

        If no chunk scores at least ``threshold``, returns ``refuse`` without
        calling the model at all.
        """
        hits = [h for h in self.search(q, k=k) if h.score >= threshold]
        if not hits:
            return refuse
        context = "\n".join(f"[chunk {h.index}] {h.text}" for h in hits)
        prompt = (
            "Answer the question using only the context below. After the answer, cite "
            "the chunk you used like [source: chunk 3]. If the context does not contain "
            f"the answer, say you don't know.\n\nContext:\n{context}\n\nQuestion: {q}"
        )
        messages = [SystemMessage(system)] if system else []
        messages.append(HumanMessage(prompt))
        return complete(messages, max_tokens=400).text

    def __len__(self):
        return len(self.chunks)

    def __repr__(self):
        return f"<ai.Index with {len(self)} chunks>"
