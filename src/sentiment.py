"""Sentiment analysis module using calibrated VADER with domain lexicon tuning, confidence scoring, and limitation tracking."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import (
    SENTIMENT_NEGATIVE,
    SENTIMENT_NEUTRAL,
    SENTIMENT_POSITIVE,
)

# Domain-specific lexicon tuning for customer product and tech reviews
CUSTOMER_REVIEW_LEXICON = {
    # Product defect & complaint terminology
    "drain": -2.0,
    "drains": -2.2,
    "draining": -2.0,
    "lag": -2.0,
    "lags": -2.0,
    "laggy": -2.2,
    "crash": -2.5,
    "crashes": -2.5,
    "crashing": -2.5,
    "unresponsive": -2.5,
    "overheat": -2.2,
    "overheats": -2.4,
    "overheating": -2.5,
    "dies": -2.0,
    "dying": -2.0,
    "unusable": -2.8,
    "stuttering": -2.0,
    "dropouts": -2.0,
    "unpair": -2.0,
    "unpairs": -2.0,
    "cracked": -2.2,
    "wobble": -1.8,
    "wobbling": -2.0,
    "wobbly": -2.0,
    "squeak": -1.5,
    "squeaking": -1.8,
    "overcharge": -2.5,
    "overcharged": -2.5,
    "dismissive": -2.2,
    "buggy": -2.2,
    "delay": -2.0,
    "delayed": -2.2,
    "delays": -2.0,
    "late": -1.8,
    "missing": -1.8,
    # High-praise satisfaction terminology
    "flawless": +2.8,
    "flawlessly": +2.8,
    "incredible": +2.8,
    "exceeded": +2.4,
    "exceeds": +2.4,
    "superb": +2.6,
}


@dataclass
class SentimentPrediction:
    """Structured result of a single sentiment prediction."""
    label: str               # "Positive", "Neutral", or "Negative"
    score: float             # Confidence score between 0.0 and 1.0
    compound: float          # VADER compound score between -1.0 and 1.0
    pos: float               # Positive proportion
    neu: float               # Neutral proportion
    neg: float               # Negative proportion
    summary_note: str        # Explanation of prediction and potential limitations


class SentimentAnalyzer:
    """
    Robust 3-class sentiment analyzer tuned for customer reviews and feedback.
    Uses calibrated VADER (Valence Aware Dictionary and sEntiment Reasoner),
    augmented with customer feedback and product review vocabulary.
    """

    def __init__(
        self,
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
    ):
        """
        Initializes the analyzer with customizable compound score cutoffs and domain lexicon.
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.analyzer = SentimentIntensityAnalyzer()
        # Calibrate with domain-specific customer review terms
        self.analyzer.lexicon.update(CUSTOMER_REVIEW_LEXICON)

    def predict_text(self, text: str) -> SentimentPrediction:
        """
        Analyzes a single text string and returns a structured SentimentPrediction.
        """
        if not text or not text.strip():
            return SentimentPrediction(
                label=SENTIMENT_NEUTRAL,
                score=0.50,
                compound=0.0,
                pos=0.0,
                neu=1.0,
                neg=0.0,
                summary_note="Empty or whitespace text defaulting to neutral.",
            )

        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]
        pos = scores["pos"]
        neu = scores["neu"]
        neg = scores["neg"]

        # Check for explicit overall verdict patterns that override isolated clause shifts
        lower_text = text.lower()
        overall_neg_patterns = [
            "bad product", "terrible product", "awful product", "horrible product",
            "poor product", "waste of money", "do not buy", "don't buy",
            "would not recommend", "not recommend", "stay away", "bad overall",
            "terrible overall", "poor overall", "overall bad", "overall poor",
            "overall terrible", "overall disappointing", "disappointing overall",
        ]
        overall_pos_patterns = [
            "great product", "excellent product", "amazing product", "best product",
            "love this product", "highly recommend", "great overall", "overall great",
            "overall excellent", "overall fantastic", "fantastic overall",
        ]

        neg_verdict_match = next((p for p in overall_neg_patterns if p in lower_text), None)
        pos_verdict_match = next((p for p in overall_pos_patterns if p in lower_text), None)

        if neg_verdict_match and not pos_verdict_match:
            label = SENTIMENT_NEGATIVE
            compound = min(-0.25, -abs(compound) if compound > 0 else compound)
            confidence = min(1.0, 0.65 + abs(compound) / 2.0)
            notes = [f"Overall negative product verdict ('{neg_verdict_match}') overrides positive sub-features."]
        elif pos_verdict_match and not neg_verdict_match:
            label = SENTIMENT_POSITIVE
            compound = max(0.25, abs(compound) if compound < 0 else compound)
            confidence = min(1.0, 0.65 + compound / 2.0)
            notes = [f"Overall positive product verdict ('{pos_verdict_match}') overrides minor negative sub-features."]
        else:
            # Determine 3-class label based on calibrated thresholds
            if compound >= self.positive_threshold:
                label = SENTIMENT_POSITIVE
                confidence = min(1.0, 0.5 + (compound / 2.0))
            elif compound <= self.negative_threshold:
                label = SENTIMENT_NEGATIVE
                confidence = min(1.0, 0.5 + (abs(compound) / 2.0))
            else:
                label = SENTIMENT_NEUTRAL
                confidence = max(0.5, neu - abs(pos - neg))
            notes = []

        if "but" in lower_text or "however" in lower_text or "although" in lower_text:
            notes.append("Mixed review phrasing detected (contains contrastive conjunction).")
        if label == SENTIMENT_NEGATIVE and neg > 0:
            notes.append(f"Negative polarity dominant (neg: {neg:.2f}, compound: {compound:.2f})")
        elif label == SENTIMENT_POSITIVE and pos > 0:
            notes.append(f"Positive polarity dominant (pos: {pos:.2f}, compound: {compound:.2f})")
        else:
            notes.append(f"Neutral tone dominance (neu: {neu:.2f})")

        summary_note = " | ".join(notes) if notes else "Standard polarity classification."

        return SentimentPrediction(
            label=label,
            score=round(float(confidence), 3),
            compound=round(float(compound), 3),
            pos=round(float(pos), 3),
            neu=round(float(neu), 3),
            neg=round(float(neg), 3),
            summary_note=summary_note,
        )

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "review_text",
    ) -> pd.DataFrame:
        """
        Runs sentiment analysis across all reviews in a DataFrame and returns an enriched DataFrame.
        """
        if df.empty:
            result_df = df.copy()
            result_df["sentiment"] = []
            result_df["sentiment_score"] = []
            result_df["sentiment_compound"] = []
            result_df["sentiment_note"] = []
            return result_df

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        predictions = [self.predict_text(text) for text in df[text_column]]

        enriched_df = df.copy()
        enriched_df["sentiment"] = [p.label for p in predictions]
        enriched_df["sentiment_score"] = [p.score for p in predictions]
        enriched_df["sentiment_compound"] = [p.compound for p in predictions]
        enriched_df["sentiment_note"] = [p.summary_note for p in predictions]

        return enriched_df

    @staticmethod
    def get_model_metadata() -> Dict[str, Any]:
        """
        Returns model specifications, strengths, and documented limitations.
        """
        return {
            "model_name": "Calibrated VADER (Lexicon & Rule-Based Sentiment Engine)",
            "classification_classes": [SENTIMENT_POSITIVE, SENTIMENT_NEUTRAL, SENTIMENT_NEGATIVE],
            "strengths": [
                "Runs 100% locally and executes in milliseconds with zero GPU or cloud requirements.",
                "Calibrated with domain-specific customer complaint terminology (draining, crashing, overheating, unpair).",
                "Explicitly handles punctuation emphasis ('terrible!!'), caps ('VERY bad'), and negation ('not good').",
                "Applies sentiment shift for contrastive clauses ('The screen is great, BUT battery life is terrible').",
                "Deterministic, fully transparent, and explainable.",
            ],
            "limitations": [
                "Complex irony or sarcasm can deceive the lexicon (e.g., 'Oh great, another crash').",
                "Subtle colloquialisms without clear polarity indicators can be scored as Neutral.",
                "Reviews with identical volumes of positive and negative praise will balance to Neutral.",
            ],
        }
