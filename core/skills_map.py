from collections import Counter
import spacy


def load_spacy_model():
    return spacy.load("en_core_web_sm")


def compute_keyword_freqs(nlp, text: str, top_n: int = 30) -> dict:
    if not text:
        return {}
    doc = nlp(text)
    words = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha
        and not token.is_stop
        and token.pos_ in {"NOUN", "PROPN", "ADJ"}
    ]
    freqs = Counter(words)
    return dict(freqs.most_common(top_n))


def make_wordcloud_html(freqs: dict, min_font: int = 14, max_font: int = 40) -> str:
    if not freqs:
        return "<p>No keywords found.</p>"

    min_f = min(freqs.values())
    max_f = max(freqs.values())

    spans = []
    for word, freq in freqs.items():
        if max_f == min_f:
            size = (min_font + max_font) / 2
        else:
            size = min_font + (freq - min_f) / (max_f - min_f) * (max_font - min_font)
        spans.append(
            f"<span style='font-size:{size:.0f}px; margin:4px; display:inline-block;'>{word}</span>"
        )

    return (
        "<div style='padding:10px; border-radius:8px; border:1px solid #444; "
        "min-height:80px;'>"
        + " ".join(spans)
        + "</div>"
    )
