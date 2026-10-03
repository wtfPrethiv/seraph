from fakes import make_chunk
from seraph.retrieval.lexical import LexicalRetriever
from seraph.retrieval.tokenize import split_identifier, tokenize
from seraph.types import AnalyzedQuery


def test_split_identifier():
    assert split_identifier("parseHTTPResponse_v2") == ["parse", "http", "response", "v", "2"]
    assert split_identifier("read_file") == ["read", "file"]


def test_tokenize_keeps_identifier_and_digits():
    toks = tokenize("def readFile(n=10**5): pass", stem=False)
    assert "readfile" in toks and "read" in toks and "file" in toks and "10" in toks


def test_bm25_ranks_identifier_match(tmp_path):
    chunks = [
        make_chunk("def parse_config(path): return json.load(open(path))"),
        make_chunk("def add(a, b): return a + b"),
        make_chunk("class HttpClient:\n    def send_request(self): ..."),
    ]
    r = LexicalRetriever()
    r.index(chunks)
    hits = r.search(AnalyzedQuery.plain("how is the config parsed"), 3)
    assert hits[0].chunk.chunk_hash == chunks[0].chunk_hash
    assert "lexical" in hits[0].scores
    r.save(tmp_path / "bm25.pkl")
    r2 = LexicalRetriever.load(tmp_path / "bm25.pkl")
    assert r2.search(AnalyzedQuery.plain("send http request"), 1)[0].chunk == chunks[2]
