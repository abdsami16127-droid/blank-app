"""Unit tests for Firebase Cloud Firestore manager and fallback modes."""

from src.firebase_manager import FirebaseManager


def test_firebase_manager_graceful_offline_fallback():
    """Verify that FirebaseManager initializes without errors when no credentials are provided."""
    fb = FirebaseManager(credentials_path="non_existent_file.json")
    status = fb.get_status()

    assert status["connected"] is False
    assert "Offline" in status["mode"]
    assert fb.is_connected is False


def test_firebase_manager_safe_noop_methods():
    """Verify that Firebase operations return safe defaults without raising exceptions in offline mode."""
    fb = FirebaseManager(credentials_path="non_existent_file.json")

    # Saving runs should return False cleanly
    saved = fb.save_analysis_run(
        dataset_name="test",
        total_reviews=10,
        total_aspects=12,
        sentiment_breakdown={"Positive": 5, "Neutral": 2, "Negative": 3},
        issue_groups=[],
        execution_time=0.1,
        throughput=100.0,
    )
    assert saved is False

    # Historical runs should return an empty list
    history = fb.get_historical_runs()
    assert history == []

    # Custom labels save/load should return False / empty dict
    assert fb.save_custom_labels({0: "Test"}) is False
    assert fb.load_custom_labels() == {}
