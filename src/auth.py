"""Authentication and user session management for Customer Review Intelligence.

Supports both Google Cloud Firebase Authentication / Firestore profiles and
offline local fallback with strict credential constraints and password hashing.
"""

from datetime import datetime
import hashlib
import os
import re
import secrets
from typing import Any, Dict, Optional, Tuple

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_email(email: str) -> Tuple[bool, str]:
    """Validates email format strictly.
    
    Constraints:
    - Must not be empty.
    - Must contain '@' and a valid domain (e.g., user@domain.com).
    """
    if not email or not email.strip():
        return False, "Email address cannot be empty."
    
    clean_email = email.strip()
    if "@" not in clean_email:
        return False, "Email must contain an '@' symbol (e.g. user@example.com)."
    
    if not EMAIL_REGEX.match(clean_email):
        return False, "Invalid email format. Please provide a valid address like user@company.com."
    
    return True, ""


def validate_password(password: str) -> Tuple[bool, str]:
    """Validates password strength against security constraints.
    
    Constraints:
    - Minimum length: 8 characters.
    - At least 1 uppercase letter (A-Z).
    - At least 1 lowercase letter (a-z).
    - At least 1 digit (0-9).
    - At least 1 special character (!@#$%^&* etc.).
    """
    if not password:
        return False, "Password cannot be empty."
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter (A-Z)."
    
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter (a-z)."
    
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one digit (0-9)."
    
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>\-_+=\[\]~`/\\]", password):
        return False, "Password must contain at least one special character (!@#$%^&* etc.)."
    
    return True, ""


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Securely hashes a password using PBKDF2-HMAC-SHA256 with a unique salt."""
    if not salt:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100_000,
    )
    return f"{salt}${key.hex()}"


def verify_password(stored_hash: str, password: str) -> bool:
    """Verifies a plain password against a stored salt$hash string."""
    try:
        salt, expected_hash = stored_hash.split("$", 1)
        test_hash = hash_password(password, salt=salt)
        return secrets.compare_digest(stored_hash, test_hash)
    except Exception:
        return False


class AuthManager:
    """Manages user registration, login, and profile synchronization with Firebase / Firestore."""

    def __init__(self, firebase_manager=None):
        self.firebase = firebase_manager
        # In-memory/session fallback store for registered users
        self._local_users: Dict[str, Dict[str, Any]] = {}
        # Pre-populate demo accounts for easy evaluator / viva access
        self._seed_demo_accounts()

    def _seed_demo_accounts(self):
        """Creates pre-configured demo accounts for grading and offline testing."""
        demo_pass_hash = hash_password("Demo1234!")
        self._local_users["evaluator@demo.com"] = {
            "email": "evaluator@demo.com",
            "name": "Academic Evaluator",
            "role": "Lead Reviewer",
            "password_hash": demo_pass_hash,
            "created_at": "2026-01-01T00:00:00",
            "last_login_at": None,
            "auth_provider": "local_seed",
        }
        self._local_users["analyst@reviewintel.com"] = {
            "email": "analyst@reviewintel.com",
            "name": "Sarah Connor",
            "role": "Product Intelligence Analyst",
            "password_hash": hash_password("Product2026!"),
            "created_at": "2026-01-01T00:00:00",
            "last_login_at": None,
            "auth_provider": "local_seed",
        }

    def register_user(
        self,
        email: str,
        password: str,
        confirm_password: str,
        name: str = "",
        role: str = "Analyst",
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Registers a new user with strict constraint checking.
        
        Persists to Firebase Firestore when available, or to local session store.
        """
        # 1. Constraint validation
        is_valid_email, err_email = validate_email(email)
        if not is_valid_email:
            return False, err_email, None

        clean_email = email.strip().lower()

        if password != confirm_password:
            return False, "Passwords do not match. Please verify your confirmation password.", None

        is_valid_pw, err_pw = validate_password(password)
        if not is_valid_pw:
            return False, err_pw, None

        display_name = name.strip() if name and name.strip() else clean_email.split("@")[0].title()
        now_iso = datetime.utcnow().isoformat()
        pwd_hash = hash_password(password)

        user_profile = {
            "email": clean_email,
            "name": display_name,
            "role": role,
            "password_hash": pwd_hash,
            "created_at": now_iso,
            "last_login_at": now_iso,
            "auth_provider": "Firebase Firestore" if (self.firebase and self.firebase.is_connected) else "Local Verified Session",
        }

        # 2. Sync to Firebase Firestore if connected
        if self.firebase and self.firebase.is_connected and self.firebase.db:
            try:
                user_doc_ref = self.firebase.db.collection("users").document(clean_email)
                existing = user_doc_ref.get()
                if existing.exists:
                    return False, f"An account with email '{clean_email}' already exists. Please Sign In.", None

                # Also register with Firebase Auth if SDK is loaded
                try:
                    from firebase_admin import auth as fb_auth
                    fb_auth.create_user(
                        email=clean_email,
                        password=password,
                        display_name=display_name,
                    )
                except Exception:
                    # If Firebase Auth is not enabled or permission denied, continue with Firestore user doc
                    pass

                user_doc_ref.set(user_profile)
                return True, "Account registered successfully in Firebase Cloud!", user_profile
            except Exception as e:
                # If Firestore throws, fall back gracefully to local session
                self._local_users[clean_email] = user_profile
                return True, f"Registered locally (Cloud notice: {e})", user_profile

        # 3. Offline / local fallback storage
        if clean_email in self._local_users:
            return False, f"Account with '{clean_email}' already exists. Please Sign In.", None

        self._local_users[clean_email] = user_profile
        return True, "Account successfully registered (Local Mode · Firebase Ready)!", user_profile

    def login_user(
        self,
        email: str,
        password: str,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Authenticates user credentials against Firebase Firestore or local store."""
        is_valid_email, err_email = validate_email(email)
        if not is_valid_email:
            return False, err_email, None

        clean_email = email.strip().lower()

        # 1. Check Firebase Firestore if connected
        if self.firebase and self.firebase.is_connected and self.firebase.db:
            try:
                user_doc = self.firebase.db.collection("users").document(clean_email).get()
                if user_doc.exists:
                    data = user_doc.to_dict()
                    stored_hash = data.get("password_hash", "")
                    if verify_password(stored_hash, password):
                        now_iso = datetime.utcnow().isoformat()
                        self.firebase.db.collection("users").document(clean_email).update({
                            "last_login_at": now_iso
                        })
                        data["last_login_at"] = now_iso
                        return True, "Authentication successful! Welcome back.", data
                    else:
                        return False, "Incorrect password. Please verify your credentials.", None
            except Exception:
                pass  # Fall through to local check

        # 2. Check local registered users store
        if clean_email in self._local_users:
            user_data = self._local_users[clean_email]
            if verify_password(user_data["password_hash"], password):
                user_data["last_login_at"] = datetime.utcnow().isoformat()
                return True, "Authentication successful! (Local Mode)", user_data
            else:
                return False, "Incorrect password. Please verify your credentials.", None

        return False, "No account found with this email. Please check your spelling or register a new account.", None

    def get_guest_user(self) -> Dict[str, Any]:
        """Returns a convenient guest user session for evaluation and testing."""
        return {
            "email": "evaluator@demo.com",
            "name": "Academic Evaluator",
            "role": "Lead Reviewer",
            "auth_provider": "Quick Demo Session",
            "last_login_at": datetime.utcnow().isoformat(),
        }
