"""Firebase Cloud Firestore backend integration for Customer Review Intelligence.

Provides persistent cloud synchronization for analysis runs, custom label overrides,
and review audit history. Includes graceful offline fallback when credentials are not yet configured.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False


class FirebaseManager:
    """Manages Firebase Firestore operations with zero-crash offline fallback."""

    def __init__(self, credentials_path: Optional[str] = None):
        self.is_connected = False
        self.project_id = ""
        self.db = None
        self.status_message = "Firebase not initialized"
        self._init_firebase(credentials_path)

    def _init_firebase(self, credentials_path: Optional[str] = None):
        if not FIREBASE_AVAILABLE:
            self.status_message = "firebase-admin library not installed. Running in local mode."
            return

        cred_path = None

        # 1. Explicit path parameter
        if credentials_path and os.path.exists(credentials_path):
            cred_path = credentials_path

        # 2. Project root default file
        if not cred_path:
            candidate = Path(__file__).resolve().parent.parent / "firebase_credentials.json"
            if candidate.exists():
                cred_path = str(candidate)

        # 3. Environment variable path
        if not cred_path and "FIREBASE_CREDENTIALS_PATH" in os.environ:
            env_path = os.environ["FIREBASE_CREDENTIALS_PATH"]
            if os.path.exists(env_path):
                cred_path = env_path

        # 4. Streamlit Secrets (for Streamlit Community Cloud)
        cred_dict = None
        if not cred_path:
            try:
                import streamlit as st
                if "firebase_credentials" in st.secrets:
                    cred_dict = dict(st.secrets["firebase_credentials"])
                elif "firebase" in st.secrets:
                    cred_dict = dict(st.secrets["firebase"])
            except Exception:
                pass

        if not cred_path and not cred_dict:
            self.status_message = "No firebase_credentials.json found. Running in local offline mode."
            return

        try:
            # Check if app already initialized
            if not firebase_admin._apps:
                if cred_dict:
                    cred = credentials.Certificate(cred_dict)
                else:
                    cred = credentials.Certificate(cred_path)
                app = firebase_admin.initialize_app(cred)
                self.project_id = app.project_id or "firebase-project"
            else:
                app = firebase_admin.get_app()
                self.project_id = app.project_id or "firebase-project"

            self.db = firestore.client()
            self.is_connected = True
            self.status_message = f"Connected to Cloud Firestore (Project: {self.project_id})"
        except Exception as e:
            self.is_connected = False
            self.status_message = f"Firebase initialization failed: {e}. Running in local mode."

    def get_status(self) -> Dict[str, Any]:
        """Returns the current connection status and backend mode."""
        return {
            "connected": self.is_connected,
            "project_id": self.project_id if self.is_connected else "Local Session",
            "mode": "Cloud Firestore (Active)" if self.is_connected else "Local Storage (Offline)",
            "message": self.status_message,
        }

    def save_analysis_run(
        self,
        dataset_name: str,
        total_reviews: int,
        total_aspects: int,
        sentiment_breakdown: Dict[str, int],
        issue_groups: List[Any],
        execution_time: float,
        throughput: float,
        user_email: Optional[str] = None,
    ) -> bool:
        """Saves a summary of an analysis run into Cloud Firestore."""
        if not self.is_connected or not self.db:
            return False

        try:
            issues_summary = []
            for g in issue_groups[:10]:
                if hasattr(g, "issue_name"):
                    issues_summary.append({
                        "cluster_id": getattr(g, "cluster_id", 0),
                        "issue_name": getattr(g, "issue_name", "Unknown"),
                        "priority": getattr(g, "priority_level", "Low"),
                        "volume": getattr(g, "review_count", 0),
                        "impact_score": getattr(g, "normalized_impact_score", 0.0),
                    })
                elif isinstance(g, dict):
                    issues_summary.append({
                        "cluster_id": g.get("cluster_id", 0),
                        "issue_name": g.get("issue_name", "Unknown"),
                        "priority": g.get("priority_level", "Low"),
                        "volume": g.get("volume", 0),
                        "impact_score": g.get("norm_impact", 0.0),
                    })

            doc_data = {
                "dataset_name": dataset_name,
                "timestamp": datetime.utcnow().isoformat(),
                "user_email": user_email or "anonymous_analyst",
                "total_reviews": total_reviews,
                "total_aspects": total_aspects,
                "sentiment_breakdown": sentiment_breakdown,
                "issue_count": len(issue_groups),
                "top_issues": issues_summary,
                "execution_time_seconds": round(execution_time, 4),
                "throughput_reviews_per_sec": round(throughput, 1),
            }

            self.db.collection("analysis_runs").add(doc_data)
            return True
        except Exception:
            return False

    def get_historical_runs(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieves recent analysis runs from Cloud Firestore."""
        if not self.is_connected or not self.db:
            return []

        try:
            docs = (
                self.db.collection("analysis_runs")
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(limit)
                .stream()
            )
            history = []
            for d in docs:
                data = d.to_dict()
                data["id"] = d.id
                history.append(data)
            return history
        except Exception:
            return []

    def save_custom_labels(self, custom_labels: Dict[int, str]) -> bool:
        """Persists human-in-the-loop custom label overrides in Cloud Firestore."""
        if not self.is_connected or not self.db:
            return False

        try:
            # Firestore document keys must be strings
            payload = {str(k): v for k, v in custom_labels.items()}
            payload["_last_updated"] = datetime.utcnow().isoformat()
            self.db.collection("app_settings").document("custom_labels").set(payload)
            return True
        except Exception:
            return False

    def load_custom_labels(self) -> Dict[int, str]:
        """Loads saved custom label overrides from Cloud Firestore."""
        if not self.is_connected or not self.db:
            return {}

        try:
            doc = self.db.collection("app_settings").document("custom_labels").get()
            if doc.exists:
                data = doc.to_dict()
                result = {}
                for k, v in data.items():
                    if k != "_last_updated":
                        try:
                            result[int(k)] = str(v)
                        except ValueError:
                            pass
                return result
            return {}
        except Exception:
            return {}
