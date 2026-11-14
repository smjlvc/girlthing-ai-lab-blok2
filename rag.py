"""
rag.py — RAG za Studentskog asistenta (GirlTHing AI Lab, Blok 2)

Sve što smo napravili na Radionici 6, uz tri poboljšanja:
  1. chunking po pasusima (ne reže usred rečenice),
  2. indeks se čuva u datoteku (embeddings se računaju samo jednom),
  3. historija razgovora (za chatbot na Radionici 7).

Upotreba:
    import rag
    index = rag.load_or_build_index(client, "data", "index.npz")
    odgovor, izvori = rag.answer(client, index, "Koliko dug treba biti CV?")
"""

import hashlib
from pathlib import Path

import numpy as np

MODEL = "gpt-6-luna"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800      # približna maksimalna dužina chunka u znakovima
TOP_K = 4             # koliko chunkova ide u kontekst
MIN_SCORE = 0.25      # prag sličnosti: ispod njega smatramo da izvor nije relevantan

RAG_INSTRUKCIJE = """Ti si Studentski asistent programa GirlTHing AI Lab.
Odgovaraj ISKLJUČIVO na osnovu priloženog KONTEKSTA.
Ako odgovor nije u kontekstu, reci: "Nemam tu informaciju u svojoj bazi znanja."
Na kraju navedi izvore u obliku [naziv_datoteke].
Odgovaraj na bosanskom, kratko i jasno."""

NEMA_KONTEKSTA = "(nema relevantnih dokumenata)"


# ---------------------------------------------------------------- dokumenti
def load_documents(folder):
    """Vraća listu (naziv_datoteke, tekst) za sve .md, .txt i .pdf datoteke."""
    docs = []
    for path in sorted(Path(folder).glob("*")):
        suffix = path.suffix.lower()          # i ".PDF" i ".pdf"
        if suffix in (".md", ".txt"):
            docs.append((path.name, path.read_text(encoding="utf-8")))
        elif suffix == ".pdf":
            from pypdf import PdfReader
            text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
            docs.append((path.name, text))
    return docs


def chunk_text(text, size=CHUNK_SIZE):
    """Dijeli tekst po pasusima i spaja susjedne pasuse dok ne pređu `size` znakova.

    Pasus duži od `size` dijeli se po rečenicama.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    pieces = []
    for p in paragraphs:
        if len(p) <= size:
            pieces.append(p)
        else:
            sentence_buf = ""
            for s in p.replace("\n", " ").split(". "):
                s = s.strip()
                if not s:
                    continue
                if sentence_buf and len(sentence_buf) + len(s) + 2 > size:
                    pieces.append(sentence_buf)
                    sentence_buf = ""
                sentence_buf = f"{sentence_buf}. {s}" if sentence_buf else s
            if sentence_buf:
                pieces.append(sentence_buf)

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------- embeddings
def embed(client, texts, batch_size=100):
    """Vraća matricu embeddinga (N × 1536), normalizovanu na dužinu 1."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch_size])
        vectors.extend(d.embedding for d in resp.data)
    V = np.array(vectors, dtype=np.float32)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


# ---------------------------------------------------------------- indeks
class Index:
    """Mala vektorska baza: chunkovi, izvori i njihovi vektori."""

    def __init__(self, chunks, sources, vectors, fingerprint=""):
        self.chunks = list(chunks)
        self.sources = list(sources)
        self.vectors = vectors
        self.fingerprint = fingerprint

    def search(self, client, query, k=TOP_K):
        """Vraća k najsličnijih chunkova kao listu (sličnost, izvor, tekst)."""
        q = embed(client, [query])[0]
        scores = self.vectors @ q
        best = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.sources[i], self.chunks[i]) for i in best]

    def save(self, path):
        np.savez(path, vectors=self.vectors,
                 chunks=np.array(self.chunks, dtype=object),
                 sources=np.array(self.sources, dtype=object),
                 fingerprint=np.array(self.fingerprint))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        return cls(d["chunks"].tolist(), d["sources"].tolist(), d["vectors"],
                   str(d["fingerprint"]))


def _fingerprint(docs):
    """Otisak sadržaja dokumenata: ako se dokumenti promijene, indeks se pravi ponovo."""
    h = hashlib.sha256()
    for name, text in docs:
        h.update(name.encode())
        h.update(text.encode())
    h.update(f"{EMBED_MODEL}|{CHUNK_SIZE}".encode())
    return h.hexdigest()


def build_index(client, data_dir):
    docs = load_documents(data_dir)
    chunks, sources = [], []
    for name, text in docs:
        for c in chunk_text(text):
            chunks.append(c)
            sources.append(name)
    vectors = embed(client, chunks)
    return Index(chunks, sources, vectors, _fingerprint(docs))


def load_or_build_index(client, data_dir, index_path="index.npz"):
    """Učita indeks iz datoteke ako postoji i odgovara dokumentima, inače ga napravi i spremi."""
    index_path = Path(index_path)
    current = _fingerprint(load_documents(data_dir))
    if index_path.exists():
        index = Index.load(index_path)
        if index.fingerprint == current:
            print(f"Indeks učitan iz {index_path} ({len(index.chunks)} chunkova).")
            return index
    index = build_index(client, data_dir)
    index.save(index_path)
    print(f"Indeks napravljen i spremljen u {index_path} ({len(index.chunks)} chunkova).")
    return index


# ---------------------------------------------------------------- RAG odgovor
def build_context(hits):
    if not hits:
        return NEMA_KONTEKSTA
    return "\n\n---\n\n".join(f"[{src}]\n{text}" for _, src, text in hits)


def answer(client, index, question, history=None, k=TOP_K, min_score=MIN_SCORE):
    """Odgovara na pitanje na osnovu baze znanja.

    history: lista prethodnih poruka [{"role": "user"/"assistant", "content": "..."}]
    Vraća (odgovor, izvori), gdje su izvori nazivi datoteka korištenih u kontekstu.
    """
    hits = [h for h in index.search(client, question, k=k) if h[0] >= min_score]
    context = build_context(hits)
    sources = list(dict.fromkeys(src for _, src, _ in hits))

    messages = list(history or [])
    messages.append({"role": "user",
                     "content": f"KONTEKST:\n{context}\n\nPITANJE:\n{question}"})

    r = client.responses.create(model=MODEL, instructions=RAG_INSTRUKCIJE, input=messages)
    return r.output_text, sources
