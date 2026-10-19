"""Multi-aspect and sentence segmentation module for granular review analysis."""

import re
from typing import List
import pandas as pd

from src.config import MIN_SEGMENT_CHAR_LENGTH, MIN_SEGMENT_WORD_COUNT


# Common subjects/aspect nouns that introduce an independent clause after 'and'
CLAUSE_SUBJECT_KEYWORDS = (
    r'(?:the\s+|it\s+|my\s+|this\s+|they\s+|we\s+|i\s+|'
    r'shipping\s+|delivery\s+|courier\s+|battery\s+|charge\s+|'
    r'customer\s+|support\s+|service\s+|rep\s+|'
    r'app\s+|application\s+|software\s+|update\s+|'
    r'hardware\s+|build\s+|hinge\s+|case\s+|sound\s+|audio\s+|screen\s+)'
)

# Contrastive clause conjunctions pattern
CONTRASTIVE_PATTERN = re.compile(
    r'(?:[.!?;]|\n+|(?:,\s*(?:but|however|although|whereas|yet|while|on\s+the\s+other\s+hand))|(?:\s+(?:but|however|although|whereas|yet)\s+))',
    re.IGNORECASE,
)

# Coordinate clause pattern: splits ', and ' OR ' and ' followed by a new clause subject
COORDINATE_PATTERN = re.compile(
    rf'(?:,\s*and\s+|\s+and\s+(?={CLAUSE_SUBJECT_KEYWORDS}))',
    re.IGNORECASE,
)


def clean_segment_text(segment: str) -> str:
    """Strips leading/trailing conjunctions, punctuation, and extra whitespace."""
    s = segment.strip()
    # Strip leading conjunctions left over from splitting
    s = re.sub(r'^(?:and\s+also|and|but|however|although|whereas|yet|while|so|too)\s+', '', s, flags=re.IGNORECASE)
    # Strip leading/trailing non-alphanumeric punctuation
    s = s.strip(" ,.-;:!?\t\n")
    if s and len(s) > 1:
        # Capitalize first letter for consistent display and embedding quality
        s = s[0].upper() + s[1:]
    return s


def is_valid_segment(segment: str) -> bool:
    """Checks whether a segment meets minimum length and word count criteria."""
    if not segment:
        return False
    if len(segment) < MIN_SEGMENT_CHAR_LENGTH:
        return False
    words = [w for w in segment.split() if w.strip()]
    if len(words) < MIN_SEGMENT_WORD_COUNT:
        return False
    return True


def segment_review_text(text: str) -> List[str]:
    """
    Decomposes a single customer review into constituent aspect sentences and clauses.
    
    Approach:
    1. Splits on sentence terminators (. ! ? ; \n) and contrastive clause markers (', but', 'however', 'although').
    2. Conditionally splits on coordinate conjunctions (', and ' or 'and <new subject>') when joining distinct predicate clauses.
    3. Trims leading conjunctions and trailing punctuation.
    4. Filters out trivial fragments (< 10 chars or < 2 words).
    5. Fallback: If segmentation produces zero valid segments, returns the original cleaned text.
    """
    if not text or not str(text).strip():
        return []

    raw_text = str(text).strip()
    
    # First pass: split by sentence ends and contrastive conjunctions
    first_pass_chunks = CONTRASTIVE_PATTERN.split(raw_text)

    # Second pass: check each chunk for coordinate clauses (", and " or "and <new subject>")
    second_pass_chunks: List[str] = []
    for chunk in first_pass_chunks:
        if not chunk or not chunk.strip():
            continue
        parts = COORDINATE_PATTERN.split(chunk)
        second_pass_chunks.extend(parts)

    # Clean and filter segments
    valid_segments: List[str] = []
    for chunk in second_pass_chunks:
        cleaned = clean_segment_text(chunk)
        if is_valid_segment(cleaned):
            valid_segments.append(cleaned)

    # Fallback to original text if splitting was overly aggressive or review was a single simple clause
    if not valid_segments:
        fallback = raw_text.strip(" \t\n\r")
        if len(fallback) >= 5:
            valid_segments = [fallback]

    return valid_segments


def segment_dataframe(
    df: pd.DataFrame,
    text_column: str = "review_text",
) -> pd.DataFrame:
    """
    Expands a DataFrame of reviews into a granular DataFrame of review segments.
    Maintains full traceability back to the parent review via 'review_id'.
    
    Columns produced:
    - segment_id: Unique integer index for the segment
    - review_id: Foreign key back to original review row
    - segment_text: The isolated aspect clause or sentence
    - original_review_text: Full verbatim text of parent review
    - segment_index: 0-indexed position of segment within its review
    - total_segments_in_review: Total count of segments extracted from this review
    + Any additional parent columns (e.g. rating, product_category)
    """
    if df.empty:
        return pd.DataFrame(columns=["segment_id", "review_id", "segment_text", "original_review_text"])

    rows = []
    segment_counter = 0

    for _, row in df.iterrows():
        r_id = row.get("review_id", _)
        orig_text = str(row[text_column])
        segments = segment_review_text(orig_text)

        total_segs = len(segments)
        for seg_idx, seg in enumerate(segments):
            seg_row = {
                "segment_id": segment_counter,
                "review_id": r_id,
                "segment_text": seg,
                "original_review_text": orig_text,
                "segment_index": seg_idx,
                "total_segments_in_review": total_segs,
            }
            # Copy over optional metadata columns from parent review
            for col in df.columns:
                if col not in [text_column, "review_id"] and col not in seg_row:
                    seg_row[col] = row[col]

            rows.append(seg_row)
            segment_counter += 1

    segments_df = pd.DataFrame(rows)
    return segments_df
