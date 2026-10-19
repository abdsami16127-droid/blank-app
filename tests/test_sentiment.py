"""Unit tests for the sentiment analysis module."""

import pytest
import pandas as pd

from src.config import (
    SENTIMENT_NEGATIVE,
    SENTIMENT_NEUTRAL,
    SENTIMENT_POSITIVE,
)
from src.sentiment import SentimentAnalyzer, SentimentPrediction


@pytest.fixture
def analyzer():
    return SentimentAnalyzer()


def test_positive_reviews(analyzer):
    """Verify clearly positive reviews are classified as Positive with high confidence."""
    review1 = "This is an incredible product, works flawlessly and exceeded all my expectations!"
    pred1 = analyzer.predict_text(review1)
    assert pred1.label == SENTIMENT_POSITIVE
    assert pred1.compound > 0.5
    assert pred1.score >= 0.70

    review2 = "Super happy with this purchase, highly recommend to everyone."
    pred2 = analyzer.predict_text(review2)
    assert pred2.label == SENTIMENT_POSITIVE
    assert pred2.compound > 0.4


def test_negative_reviews(analyzer):
    """Verify customer complaints and failure reports are classified as Negative."""
    review1 = "The battery drains in less than two hours, totally unusable and terrible quality."
    pred1 = analyzer.predict_text(review1)
    assert pred1.label == SENTIMENT_NEGATIVE
    assert pred1.compound < -0.4
    assert pred1.score >= 0.70

    review2 = "Customer service was completely dismissive and refused to refund my money."
    pred2 = analyzer.predict_text(review2)
    assert pred2.label == SENTIMENT_NEGATIVE
    assert pred2.compound < -0.3


def test_neutral_reviews(analyzer):
    """Verify purely factual or neutral observations are classified as Neutral."""
    review1 = "The package arrived on Tuesday in a standard brown cardboard box."
    pred1 = analyzer.predict_text(review1)
    assert pred1.label == SENTIMENT_NEUTRAL
    assert -0.05 <= pred1.compound <= 0.05

    review2 = "Item has standard dimensions and works according to the basic specification sheet."
    pred2 = analyzer.predict_text(review2)
    assert pred2.label == SENTIMENT_NEUTRAL


def test_negation_handling(analyzer):
    """Verify that negations (e.g. 'not good') correctly invert polarity to Negative."""
    pred = analyzer.predict_text("This product is not good and not worth the price.")
    assert pred.label == SENTIMENT_NEGATIVE
    assert pred.compound < 0


def test_contrastive_conjunction_notes(analyzer):
    """Verify that sentences with 'but' trigger a mixed review note and proper sentiment shift."""
    pred = analyzer.predict_text("The screen is bright, but the battery dies in two hours.")
    assert "Mixed review phrasing detected" in pred.summary_note
    # Calibrated 'dies' pulls the complaint negative
    assert pred.label == SENTIMENT_NEGATIVE


def test_analyze_dataframe(analyzer):
    """Verify batch DataFrame sentiment enrichment."""
    data = {
        "review_id": [0, 1, 2],
        "review_text": [
            "Great build quality, very solid.",
            "Item arrived on Tuesday in standard packaging.",
            "Crashes all the time, terrible bug.",
        ],
    }
    df = pd.DataFrame(data)
    enriched = analyzer.analyze_dataframe(df)

    assert "sentiment" in enriched.columns
    assert "sentiment_score" in enriched.columns
    assert "sentiment_compound" in enriched.columns
    assert "sentiment_note" in enriched.columns

    assert enriched["sentiment"].iloc[0] == SENTIMENT_POSITIVE
    assert enriched["sentiment"].iloc[1] == SENTIMENT_NEUTRAL
    assert enriched["sentiment"].iloc[2] == SENTIMENT_NEGATIVE


def test_empty_text_safety(analyzer):
    """Verify that empty strings or whitespace don't crash and yield neutral defaults."""
    pred = analyzer.predict_text("   ")
    assert pred.label == SENTIMENT_NEUTRAL
    assert pred.score == 0.50


def test_model_metadata():
    """Verify that model limitations and capabilities are explicitly documented."""
    metadata = SentimentAnalyzer.get_model_metadata()
    assert "model_name" in metadata
    assert len(metadata["classification_classes"]) == 3
    assert len(metadata["strengths"]) > 0
    assert len(metadata["limitations"]) > 0
