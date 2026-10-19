"""Unit tests for user authentication and constraint validation."""

import pytest
from src.auth import (
    AuthManager,
    hash_password,
    validate_email,
    validate_password,
    verify_password,
)


def test_email_validation():
    # Invalid emails
    assert not validate_email("")[0]
    assert not validate_email("plainaddress")[0]
    assert not validate_email("missing_at_symbol.com")[0]
    assert not validate_email("@no_local_part.com")[0]
    assert not validate_email("user@no_domain")[0]

    # Valid emails
    assert validate_email("user@example.com")[0]
    assert validate_email("student.first@college.edu")[0]
    assert validate_email("analyst_123@enterprise.org")[0]


def test_password_validation_constraints():
    # Empty
    assert not validate_password("")[0]

    # Length < 8
    is_valid, msg = validate_password("Short1!")
    assert not is_valid
    assert "8 characters" in msg

    # Missing uppercase
    is_valid, msg = validate_password("lowercase123!@#")
    assert not is_valid
    assert "uppercase" in msg

    # Missing lowercase
    is_valid, msg = validate_password("UPPERCASE123!@#")
    assert not is_valid
    assert "lowercase" in msg

    # Missing digit
    is_valid, msg = validate_password("NoDigitsHere!@#")
    assert not is_valid
    assert "digit" in msg

    # Missing special character
    is_valid, msg = validate_password("NoSpecial1234")
    assert not is_valid
    assert "special character" in msg

    # Valid strong password
    assert validate_password("ValidPassword2026!")[0]
    assert validate_password("Secret#Pass99")[0]


def test_password_hashing_and_verification():
    raw_pass = "SecurePass123$"
    hashed = hash_password(raw_pass)
    assert hashed != raw_pass
    assert "$" in hashed
    assert verify_password(hashed, raw_pass)
    assert not verify_password(hashed, "WrongPass123$")


def test_auth_manager_registration_and_login():
    auth_mgr = AuthManager(firebase_manager=None)

    # 1. Registration with mismatched passwords
    ok, msg, user = auth_mgr.register_user(
        email="test@enterprise.com",
        password="ValidPassword2026!",
        confirm_password="DifferentPassword2026!",
    )
    assert not ok
    assert "do not match" in msg

    # 2. Registration with invalid password
    ok, msg, user = auth_mgr.register_user(
        email="test@enterprise.com",
        password="weak",
        confirm_password="weak",
    )
    assert not ok

    # 3. Successful registration
    ok, msg, user = auth_mgr.register_user(
        email="test@enterprise.com",
        password="ValidPassword2026!",
        confirm_password="ValidPassword2026!",
        name="Alex Mercer",
        role="Product Manager",
    )
    assert ok
    assert user is not None
    assert user["email"] == "test@enterprise.com"
    assert user["name"] == "Alex Mercer"

    # 4. Duplicate registration rejected
    ok2, msg2, _ = auth_mgr.register_user(
        email="test@enterprise.com",
        password="ValidPassword2026!",
        confirm_password="ValidPassword2026!",
    )
    assert not ok2
    assert "already exists" in msg2

    # 5. Successful login
    login_ok, login_msg, logged_user = auth_mgr.login_user(
        email="test@enterprise.com",
        password="ValidPassword2026!",
    )
    assert login_ok
    assert logged_user["email"] == "test@enterprise.com"

    # 6. Wrong password login
    wrong_ok, wrong_msg, _ = auth_mgr.login_user(
        email="test@enterprise.com",
        password="WrongPassword2026!",
    )
    assert not wrong_ok
    assert "Incorrect password" in wrong_msg

    # 7. Non-existent user login
    none_ok, none_msg, _ = auth_mgr.login_user(
        email="unknown@enterprise.com",
        password="ValidPassword2026!",
    )
    assert not none_ok
    assert "No account found" in none_msg


def test_seed_demo_accounts_and_guest_user():
    auth_mgr = AuthManager(firebase_manager=None)
    # Demo seed user
    ok, _, user = auth_mgr.login_user("evaluator@demo.com", "Demo1234!")
    assert ok
    assert user["name"] == "Academic Evaluator"

    # Quick guest user
    guest = auth_mgr.get_guest_user()
    assert guest["email"] == "evaluator@demo.com"
