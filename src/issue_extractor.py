"""Topic extraction, c-TF-IDF keyword analysis, and Actionable Issue Prioritization."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.clustering import SemanticClusterer
from src.config import (
    NOISE_CLUSTER_LABEL,
    NOISE_CLUSTER_NAME,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PRIORITY_THRESHOLDS,
    SENTIMENT_NEGATIVE,
    SENTIMENT_NEUTRAL,
    SENTIMENT_POSITIVE,
)


@dataclass
class IssueGroup:
    """Summary of a single identified issue group with priority scoring."""
    cluster_id: int
    issue_name: str
    review_count: int               # Total reviews/segments assigned
    percentage: float               # Percentage of corpus
    sentiment_breakdown: Dict[str, int]
    representative_review: str
    top_keywords: List[str]
    member_review_ids: List[int]
    # Milestone 6 Priority Metrics
    negative_sentiment_ratio: float = 0.0     # Negative count / review_count (0.0 to 1.0)
    raw_impact_score: float = 0.0             # Volume * Negative Ratio (= Negative Count)
    normalized_impact_score: float = 0.0      # (Raw Impact / Total Dataset Items) * 100
    priority_level: str = PRIORITY_LOW        # "Critical", "High", "Medium", "Low"
    priority_rationale: str = ""


def format_issue_title(keywords: List[str], fallback_text: str = "") -> str:
    """
    Constructs a readable, professional title from top c-TF-IDF keywords.
    Capitalizes keywords and combines the top 2-3 terms.
    """
    if not keywords:
        if fallback_text:
            words = fallback_text.split()[:5]
            return " ".join(words).title()
        return "General Issue"

    clean_kws = [kw.strip().title() for kw in keywords if kw.strip()]
    if not clean_kws:
        return "General Issue"

    if len(clean_kws) == 1:
        return clean_kws[0]
    elif len(clean_kws) == 2:
        return f"{clean_kws[0]} / {clean_kws[1]}"
    else:
        return f"{clean_kws[0]} / {clean_kws[1]}"


class IssueExtractor:
    """
    Extracts descriptive issue names, representative customer quotes,
    and calculates actionable issue priority indices using c-TF-IDF and sentiment weighting.
    """

    def __init__(self, stop_words: str = "english", ngram_range: tuple = (1, 2)):
        self.vectorizer = CountVectorizer(
            stop_words=stop_words,
            ngram_range=ngram_range,
            min_df=1,
            max_features=2000,
        )

    def extract_issues(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        text_column: str = "review_text",
        top_n_keywords: int = 5,
    ) -> List[IssueGroup]:
        """
        Processes clustered reviews and returns a list of ranked IssueGroup objects
        with calculated Impact Scores and Priority Levels.
        """
        if df.empty or len(cluster_labels) == 0:
            return []

        medoids = SemanticClusterer.find_cluster_medoids(embeddings, cluster_labels)
        unique_clusters = sorted(set(cluster_labels))
        total_reviews = len(df)

        # 1. c-TF-IDF keyword extraction: aggregate text per cluster
        cluster_docs: Dict[int, str] = {}
        for c_id in unique_clusters:
            mask = cluster_labels == c_id
            cluster_text = " ".join(df.loc[mask, text_column].astype(str).tolist())
            cluster_docs[c_id] = cluster_text

        # Fit TF-IDF across the aggregated cluster documents
        ordered_c_ids = [c for c in unique_clusters if c != NOISE_CLUSTER_LABEL]
        cluster_keywords: Dict[int, List[str]] = {NOISE_CLUSTER_LABEL: []}

        if ordered_c_ids:
            docs = [cluster_docs[c] for c in ordered_c_ids]
            try:
                tfidf_matrix = self.vectorizer.fit_transform(docs)
                feature_names = self.vectorizer.get_feature_names_out()

                for row_idx, c_id in enumerate(ordered_c_ids):
                    row_scores = tfidf_matrix[row_idx].toarray().flatten()
                    top_indices = np.argsort(row_scores)[::-1][:top_n_keywords]
                    top_words = [
                        feature_names[i] for i in top_indices if row_scores[i] > 0
                    ]
                    cluster_keywords[c_id] = top_words
            except Exception:
                # If vocabulary is empty (e.g. all stopwords)
                for c_id in ordered_c_ids:
                    cluster_keywords[c_id] = []

        # 2. Build IssueGroup objects with Priority Metrics
        issue_groups: List[IssueGroup] = []
        for c_id in unique_clusters:
            mask = cluster_labels == c_id
            member_indices = np.where(mask)[0]
            count = int(len(member_indices))
            percentage = round((count / total_reviews) * 100.0, 1)

            # Representative medoid review
            medoid_idx = medoids.get(c_id, member_indices[0])
            representative_quote = str(df.iloc[medoid_idx][text_column])

            # Sentiment breakdown
            sentiment_counts = {
                SENTIMENT_POSITIVE: 0,
                SENTIMENT_NEUTRAL: 0,
                SENTIMENT_NEGATIVE: 0,
            }
            if "sentiment" in df.columns:
                cluster_sentiments = df.loc[mask, "sentiment"].value_counts().to_dict()
                for s_label in sentiment_counts:
                    sentiment_counts[s_label] = int(cluster_sentiments.get(s_label, 0))

            # Impact Score Calculation:
            # Impact Score = Volume * Negative Sentiment Ratio
            neg_count = sentiment_counts[SENTIMENT_NEGATIVE]
            neg_ratio = round(neg_count / count, 3) if count > 0 else 0.0
            raw_impact = round(count * neg_ratio, 2)
            # Normalized Impact Score: % of entire feedback corpus that is a negative complaint in this group
            norm_impact = round((raw_impact / total_reviews) * 100.0, 1) if total_reviews > 0 else 0.0

            # Priority Level Assignment based on transparent thresholds
            if norm_impact >= PRIORITY_THRESHOLDS[PRIORITY_CRITICAL]:
                priority_lvl = PRIORITY_CRITICAL
                priority_note = (
                    f"Concentrated customer friction: represents {norm_impact:.1f}% of total corpus "
                    f"with {neg_ratio:.0%} negative sentiment ratio."
                )
            elif norm_impact >= PRIORITY_THRESHOLDS[PRIORITY_HIGH]:
                priority_lvl = PRIORITY_HIGH
                priority_note = (
                    f"Substantial dissatisfaction: represents {norm_impact:.1f}% of total corpus "
                    f"with {neg_ratio:.0%} negative sentiment ratio."
                )
            elif norm_impact >= PRIORITY_THRESHOLDS[PRIORITY_MEDIUM]:
                priority_lvl = PRIORITY_MEDIUM
                priority_note = (
                    f"Moderate friction: represents {norm_impact:.1f}% of total corpus "
                    f"with {neg_ratio:.0%} negative sentiment ratio."
                )
            else:
                priority_lvl = PRIORITY_LOW
                priority_note = (
                    f"Low friction or predominantly positive/neutral feedback ({norm_impact:.1f}% negative impact)."
                )

            # Naming
            if c_id == NOISE_CLUSTER_LABEL:
                issue_name = NOISE_CLUSTER_NAME
                keywords = ["Uncategorized"]
            else:
                keywords = cluster_keywords.get(c_id, [])
                issue_name = format_issue_title(keywords, fallback_text=representative_quote)

            member_ids = (
                df.loc[mask, "review_id"].tolist()
                if "review_id" in df.columns
                else list(member_indices)
            )

            issue_groups.append(
                IssueGroup(
                    cluster_id=int(c_id),
                    issue_name=issue_name,
                    review_count=count,
                    percentage=percentage,
                    sentiment_breakdown=sentiment_counts,
                    representative_review=representative_quote,
                    top_keywords=keywords,
                    member_review_ids=member_ids,
                    negative_sentiment_ratio=neg_ratio,
                    raw_impact_score=raw_impact,
                    normalized_impact_score=norm_impact,
                    priority_level=priority_lvl,
                    priority_rationale=priority_note,
                )
            )

        # Sort by priority impact descending (Critical & High issues first, noise at the end)
        issue_groups.sort(
            key=lambda g: (
                g.cluster_id == NOISE_CLUSTER_LABEL,
                -g.normalized_impact_score,
                -g.review_count,
            )
        )
        return issue_groups

    @staticmethod
    def assign_issue_labels_to_dataframe(
        df: pd.DataFrame,
        issue_groups: List[IssueGroup],
        cluster_labels: np.ndarray,
    ) -> pd.DataFrame:
        """
        Enriches DataFrame with 'cluster_id', 'issue_name', 'priority_level', and 'impact_score'.
        """
        cluster_to_name = {g.cluster_id: g.issue_name for g in issue_groups}
        cluster_to_priority = {g.cluster_id: g.priority_level for g in issue_groups}
        cluster_to_impact = {g.cluster_id: g.normalized_impact_score for g in issue_groups}

        enriched = df.copy()
        enriched["cluster_id"] = cluster_labels
        enriched["issue_name"] = [
            cluster_to_name.get(cid, NOISE_CLUSTER_NAME) for cid in cluster_labels
        ]
        enriched["priority_level"] = [
            cluster_to_priority.get(cid, PRIORITY_LOW) for cid in cluster_labels
        ]
        enriched["impact_score"] = [
            cluster_to_impact.get(cid, 0.0) for cid in cluster_labels
        ]
        return enriched
