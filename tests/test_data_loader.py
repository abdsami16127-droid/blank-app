"""Unit tests for the data loader and validation module."""

import io
import pytest
import pandas as pd

from src.config import SAMPLE_CSV_PATH
from src.data_loader import (
    DataValidationError,
    clean_text_content,
    create_single_review_dataframe,
    detect_review_column,
    load_reviews_from_csv,
)


def test_clean_text_content():
    """Verify whitespace stripping, multiple space collapsing, and non-printable handling."""
    assert clean_text_content("  Great product!   ") == "Great product!"
    assert clean_text_content("Line one\n\nLine two\t") == "Line one Line two"
    assert clean_text_content(None) == ""
    assert clean_text_content(float("nan")) == ""
    # Null byte stripping
    assert clean_text_content("Clean\x00Text") == "CleanText"


def test_detect_review_column():
    """Test column detection priority and fuzzy matching."""
    cols1 = ["id", "timestamp", "review_text", "score"]
    assert detect_review_column(cols1) == "review_text"

    cols2 = ["ID", "Customer_Feedback", "Rating"]
    assert detect_review_column(cols2) == "Customer_Feedback"

    cols3 = ["content"]
    assert detect_review_column(cols3) == "content"

    # Explicit override
    assert detect_review_column(cols1, explicit_column="score") == "score"

    # Column not found
    assert detect_review_column(["order_id", "price", "quantity"]) is None


def test_load_sample_csv():
    """Test loading the realistic sample dataset."""
    assert SAMPLE_CSV_PATH.exists()
    df, report = load_reviews_from_csv(SAMPLE_CSV_PATH)

    assert report.is_valid
    assert report.total_rows_read > 50
    assert report.valid_rows == report.total_rows_read
    assert "review_id" in df.columns
    assert "review_text" in df.columns
    assert df["review_id"].iloc[0] == 0
    assert len(df["review_text"].iloc[0]) > 10


def test_load_csv_with_missing_and_short_reviews():
    """Verify that empty, whitespace, and trivial reviews are filtered with warnings recorded."""
    csv_data = """review,rating
The battery dies very quickly.,1
,1
"   ",2
Hi,3
"The screen is cracked and won't turn on.",1
"""
    buffer = io.StringIO(csv_data)
    df, report = load_reviews_from_csv(buffer)

    assert report.total_rows_read == 5
    # Row 2 & 3 are empty/whitespace -> 2 dropped
    assert report.empty_or_null_dropped == 2
    # Row 4 is "Hi" (< 5 chars) -> 1 dropped
    assert report.short_rows_dropped == 1
    # Valid rows remaining: 2
    assert report.valid_rows == 2
    assert len(df) == 2
    assert df["review_text"].iloc[0] == "The battery dies very quickly."
    assert df["review_text"].iloc[1] == "The screen is cracked and won't turn on."


def test_missing_column_raises_error():
    """A CSV with no recognizable review column must raise DataValidationError."""
    csv_data = """product_id,price,stock
101,29.99,5
102,49.99,12
"""
    buffer = io.StringIO(csv_data)
    with pytest.raises(DataValidationError) as exc_info:
        load_reviews_from_csv(buffer)
    assert "Could not find a review text column" in str(exc_info.value)


def test_empty_csv_raises_error():
    """An empty CSV must raise DataValidationError."""
    buffer = io.StringIO("")
    with pytest.raises(DataValidationError):
        load_reviews_from_csv(buffer)


def test_all_rows_invalid_raises_error():
    """If all rows are empty or too short, an error must be raised."""
    csv_data = """review
OK
a

"  "
"""
    buffer = io.StringIO(csv_data)
    with pytest.raises(DataValidationError) as exc_info:
        load_reviews_from_csv(buffer)
    assert "No valid reviews remained" in str(exc_info.value)


def test_create_single_review_dataframe():
    """Verify manual review entry creation and validation."""
    valid_text = "The audio cuts out whenever I turn my head."
    df, report = create_single_review_dataframe(valid_text)

    assert report.is_valid
    assert len(df) == 1
    assert df["review_id"].iloc[0] == 0
    assert df["review_text"].iloc[0] == valid_text
    assert df["source"].iloc[0] == "manual_entry"

    # Empty manual review
    with pytest.raises(DataValidationError):
        create_single_review_dataframe("   ")

    # Too short manual review
    with pytest.raises(DataValidationError):
        create_single_review_dataframe("Bad")


def test_load_csv_with_existing_review_id_column():
    """Verify that uploading a CSV that already has a 'review_id' column succeeds without duplicate insert error."""
    csv_data = """review_id,review,rating
REV-001,The battery drains too quickly in the cold,1
REV-002,Shipping was delayed by three whole weeks,1
"""
    buffer = io.StringIO(csv_data)
    df, report = load_reviews_from_csv(buffer)

    assert report.is_valid
    assert len(df) == 2
    assert "review_id" in df.columns
    assert df["review_id"].iloc[0] == 0
    assert df["review_id"].iloc[1] == 1
    assert "original_review_id" in df.columns
    assert df["original_review_id"].iloc[0] == "REV-001"
    assert df["original_review_id"].iloc[1] == "REV-002"

