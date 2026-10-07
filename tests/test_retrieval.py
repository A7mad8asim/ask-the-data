from askdata.resources import load_examples
from askdata.retrieval import ExampleRetriever, normalize_text


def test_arabic_normalization():
    assert normalize_text("أحمد إلى آخر") == normalize_text("احمد الي اخر")
    assert normalize_text("المتابعةُ ١٢") == "المتابعه 12"


def test_arabic_question_finds_the_arabic_pattern():
    r = ExampleRetriever(load_examples())
    top = r.top_k("كم نسبة الغياب في عيادة الريان في 2024؟", 1)
    assert top[0]["id"] == "X04"  # same question shape: no-show rate at one clinic, in Arabic


def test_english_question_finds_the_english_pattern():
    r = ExampleRetriever(load_examples())
    ids = [e["id"] for e in r.top_k("How many Dental visits did each clinic have in 2024?", 3)]
    assert "X28" in ids


def test_embedding_failure_falls_back_to_tfidf():
    def broken(texts):
        raise ConnectionError("no ollama")

    r = ExampleRetriever(load_examples(), method="bge-m3", embed=broken)
    assert r.method == "tfidf" and len(r.top_k("no-show rate", 2)) == 2
