"""Data ingestion, validation, and normalization module."""

from dataclasses import dataclass, field
import io
from pathlib import Path
import re
from typing import List, Optional, Tuple, Union

import pandas as pd

from src.config import (
    CANDIDATE_REVIEW_COLUMNS,
    MAX_REVIEW_CHAR_LENGTH,
    MIN_REVIEW_CHAR_LENGTH,
)


class DataValidationError(Exception):
    """Raised when uploaded or provided review data fails validation."""
    pass


@dataclass
class ValidationReport:
    """Summary of data ingestion and cleaning results."""
    total_rows_read: int = 0
    valid_rows: int = 0
    empty_or_null_dropped: int = 0
    short_rows_dropped: int = 0
    duplicate_rows_dropped: int = 0
    detected_column: str = ""
    warning_messages: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Returns True if there is at least one usable review."""
        return self.valid_rows > 0


def detect_review_column(
    columns: List[str], explicit_column: Optional[str] = None
) -> Optional[str]:
    """
    Detects the column name containing customer reviews.
    Prioritizes explicit user choice, then known candidate names, then fuzzy matches.
    """
    if explicit_column and explicit_column in columns:
        return explicit_column

    # Normalize existing columns for case-insensitive lookup
    col_map = {str(col).strip().lower(): str(col) for col in columns}

    # 1. Exact match with candidate list
    for candidate in CANDIDATE_REVIEW_COLUMNS:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]

    # 2. Substring match (e.g., 'user_reviews', 'feedback_text')
    for norm_name, original_name in col_map.items():
        if "review" in norm_name or "feedback" in norm_name or "comment" in norm_name:
            return original_name

    # 3. If there is only one column in the entire CSV, assume it's the text
    if len(columns) == 1:
        return columns[0]

    return None


def clean_text_content(text: Union[str, float, None]) -> str:
    """Cleans a single text string by stripping whitespace and non-printable artifacts."""
    if pd.isna(text) or text is None:
        return ""
    text_str = str(text)
    # Remove null bytes and non-printable control characters (except newline/tab)
    text_str = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text_str)
    # Normalize multiple whitespace characters
    text_str = re.sub(r"\s+", " ", text_str)
    return text_str.strip()


def load_reviews_from_csv(
    source: Union[str, Path, io.BytesIO, io.StringIO],
    explicit_column: Optional[str] = None,
    drop_duplicates: bool = False,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Loads, validates, and cleans reviews from a CSV file path or file-like buffer.

    Returns:
        Tuple of (clean_dataframe, validation_report)

    Raises:
        DataValidationError: If the file is unreadable, empty, or has no valid text column.
    """
    report = ValidationReport()

    # Determine input type & try common encodings if reading raw bytes or path
    df: Optional[pd.DataFrame] = None
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    if isinstance(source, (str, Path)):
        file_path = Path(source)
        if not file_path.exists():
            raise DataValidationError(f"File not found: {file_path}")
        if file_path.stat().st_size == 0:
            raise DataValidationError("Uploaded CSV file is completely empty.")

        for enc in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    else:
        # Buffer / BytesIO from Streamlit or tests
        try:
            # Check for empty buffer
            if hasattr(source, "getvalue") and len(source.getvalue()) == 0:
                raise DataValidationError("Uploaded file contains zero bytes.")

            # Try reading as is
            df = pd.read_csv(source)
        except UnicodeDecodeError:
            if hasattr(source, "seek"):
                source.seek(0)
            for enc in encodings_to_try[1:]:
                try:
                    df = pd.read_csv(source, encoding=enc)
                    break
                except Exception:
                    if hasattr(source, "seek"):
                        source.seek(0)
                    continue
        except pd.errors.EmptyDataError:
            raise DataValidationError("The CSV file has no columns or data.")
        except Exception as e:
            raise DataValidationError(f"Could not parse CSV file: {str(e)}")

    if df is None:
        raise DataValidationError(
            "Could not decode the CSV file. Please ensure it is saved in UTF-8 format."
        )

    report.total_rows_read = len(df)
    if report.total_rows_read == 0:
        raise DataValidationError("CSV file contains headers but no rows of data.")

    # Detect the review text column
    detected_col = detect_review_column(list(df.columns), explicit_column)
    if not detected_col:
        available_cols = ", ".join(f"'{c}'" for c in df.columns)
        raise DataValidationError(
            f"Could not find a review text column. Found columns: [{available_cols}]. "
            f"Please rename the review column to 'review' or 'review_text', or specify it manually."
        )

    report.detected_column = detected_col

    # Standardize column naming: Keep original columns, but ensure 'review_text' contains cleaned text
    working_df = df.copy()
    raw_texts = working_df[detected_col]

    # Clean text values
    cleaned_texts = raw_texts.apply(clean_text_content)

    # 1. Identify null / whitespace-only entries
    empty_mask = cleaned_texts.str.len() == 0
    report.empty_or_null_dropped = int(empty_mask.sum())
    if report.empty_or_null_dropped > 0:
        report.warning_messages.append(
            f"Ignored {report.empty_or_null_dropped} row(s) with empty or missing review text."
        )

    valid_mask = ~empty_mask
    filtered_df = working_df[valid_mask].copy()
    cleaned_texts = cleaned_texts[valid_mask]

    # 2. Identify reviews that are too short (trivial fragments)
    short_mask = cleaned_texts.str.len() < MIN_REVIEW_CHAR_LENGTH
    report.short_rows_dropped = int(short_mask.sum())
    if report.short_rows_dropped > 0:
        report.warning_messages.append(
            f"Ignored {report.short_rows_dropped} row(s) shorter than {MIN_REVIEW_CHAR_LENGTH} characters."
        )

    valid_mask = ~short_mask
    filtered_df = filtered_df[valid_mask].copy()
    cleaned_texts = cleaned_texts[valid_mask]

    # 3. Optional duplicate handling
    if drop_duplicates:
        dup_mask = cleaned_texts.duplicated(keep="first")
        report.duplicate_rows_dropped = int(dup_mask.sum())
        if report.duplicate_rows_dropped > 0:
            report.warning_messages.append(
                f"Removed {report.duplicate_rows_dropped} duplicate review(s)."
            )
            filtered_df = filtered_df[~dup_mask].copy()
            cleaned_texts = cleaned_texts[~dup_mask]

    report.valid_rows = len(filtered_df)
    if report.valid_rows == 0:
        raise DataValidationError(
            "No valid reviews remained after removing empty and trivial rows."
        )

    # Add normalized standardized columns
    filtered_df["review_text"] = cleaned_texts
    # Enforce maximum text length truncation if review is abnormally huge
    filtered_df["review_text"] = filtered_df["review_text"].apply(
        lambda t: t[:MAX_REVIEW_CHAR_LENGTH]
    )
    # Assign persistent 0-indexed ID for review traceability
    filtered_df.reset_index(drop=True, inplace=True)
    if "review_id" in filtered_df.columns:
        # If the input CSV already contained a review_id, preserve it in original_review_id
        if "original_review_id" not in filtered_df.columns:
            filtered_df["original_review_id"] = filtered_df["review_id"]
        filtered_df["review_id"] = filtered_df.index
        # Move review_id to column index 0
        cols = [c for c in filtered_df.columns if c != "review_id"]
        filtered_df = filtered_df[["review_id"] + cols]
    else:
        filtered_df.insert(0, "review_id", filtered_df.index)

    return filtered_df, report


def create_single_review_dataframe(review_input: str) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Creates a standardized review DataFrame from manual user input.
    Supports either a single review or multiple reviews (one per line).
    """
    if not review_input or not review_input.strip():
        raise DataValidationError("Review text cannot be empty or only whitespace.")

    # Split lines in case user pasted multiple reviews
    lines = [clean_text_content(l) for l in review_input.strip().split("\n")]
    valid_lines = [l for l in lines if len(l) >= MIN_REVIEW_CHAR_LENGTH]

    if not valid_lines:
        raise DataValidationError(
            f"No valid reviews found. Each review must be at least {MIN_REVIEW_CHAR_LENGTH} characters."
        )

    report = ValidationReport(
        total_rows_read=len(lines),
        valid_rows=len(valid_lines),
        short_rows_dropped=len(lines) - len(valid_lines),
        detected_column="manual_input",
    )

    rows = [
        {
            "review_id": i,
            "review_text": line[:MAX_REVIEW_CHAR_LENGTH],
            "source": "manual_entry",
        }
        for i, line in enumerate(valid_lines)
    ]
    df = pd.DataFrame(rows)
    return df, report
