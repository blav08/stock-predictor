"""Unit tests for 10-K text processing on small synthetic documents."""
import math

from stockpred.text import (
    cosine_similarity, extract_mdna, extract_section, html_to_text, jaccard_similarity,
    new_sentences, sentiment_scores, tokenize,
)

BODY = "The company faces significant competition and may lose market share. " * 60

DOC = f"""
<html><body>
<p>Table of Contents</p>
<p>Item 1A. Risk Factors</p><p>Item 1B. Unresolved Staff Comments</p><p>Item 7. MD&amp;A</p><p>Item 8.</p>
<p>Item 1A. Risk Factors</p>
<p>{BODY}</p>
<p>For details see <a href="#x">Item 1A</a> and <a href="#y">Item 2</a> in this report.</p>
<p>Item 1B. Unresolved Staff Comments</p><p>None.</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>{BODY * 3}</p>
<p>Item 8. Financial Statements</p>
</body></html>
"""


def test_inline_links_do_not_become_headings():
    text = html_to_text(DOC)
    assert "see Item 1A and Item 2 in this report" in text


def test_extract_risk_factors_skips_table_of_contents():
    section = extract_section(html_to_text(DOC), "1A", ("1B", "1C", "2"))
    assert section.startswith("Item 1A. Risk Factors")
    assert "Unresolved Staff Comments" not in section
    assert "significant competition" in section


def test_mdna_fallback_by_title():
    doc = f"<p>Management's Discussion and Analysis of Financial Condition</p><p>{BODY * 3}</p>" \
          f"<p>Report of Independent Registered Public Accounting Firm</p>"
    section = extract_mdna(html_to_text(doc))
    assert section is not None and "significant competition" in section


def test_sentiment_counts_lm_words():
    scores = sentiment_scores(tokenize("loss loss decline growth improve the the the the"))
    assert scores["n_words"] == 9
    assert scores["lm_negative"] > scores["lm_positive"] > 0
    assert -1 <= scores["net_tone"] < 0


def test_similarity_bounds():
    a = tokenize("revenue increased due to strong demand")
    assert math.isclose(cosine_similarity(a, a), 1.0)
    assert cosine_similarity(a, tokenize("completely different words here")) == 0
    assert 0 < jaccard_similarity(a, tokenize("revenue decreased due to weak demand")) < 1


def test_new_sentences_ignore_changed_numbers():
    prev = "In 2025 we had 100 employees working in our offices worldwide today."
    curr = ("In 2026 we had 120 employees working in our offices worldwide today. "
            "We are now exposed to new tariffs imposed on imports of our key components.")
    added = new_sentences(curr, prev)
    assert len(added) == 1 and "tariffs" in added[0]
