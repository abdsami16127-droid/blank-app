"""UI components for user authentication, registration modal, and session status."""

import streamlit as st
from typing import Any, Dict, Optional

from src.auth import (
    AuthManager,
    validate_email,
    validate_password,
)


def render_login_page(auth_mgr: AuthManager) -> bool:
    """Renders the enterprise login and registration portal.
    
    Returns True if the user is authenticated, False otherwise.
    """
    if st.session_state.get("authenticated", False):
        return True

    # Centered container for professional login experience
    col_l, col_center, col_r = st.columns([1, 2, 1])

    with col_center:
        st.markdown(
            """
            <div style="text-align: center; margin-top: 1rem; margin-bottom: 1.5rem;">
                <div style="display: inline-block; margin-bottom: 10px;">
                    <svg width="56" height="56" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 4px 10px rgba(79,70,229,0.25));">
                        <defs>
                            <linearGradient id="loginLogoGrad" x1="0" y1="0" x2="44" y2="44" gradientUnits="userSpaceOnUse">
                                <stop offset="0%" stop-color="#4F46E5"/>
                                <stop offset="50%" stop-color="#6366F1"/>
                                <stop offset="100%" stop-color="#06B6D4"/>
                            </linearGradient>
                        </defs>
                        <rect width="44" height="44" rx="14" fill="url(#loginLogoGrad)"/>
                        <path d="M11 28V23C11 22.4477 11.4477 22 12 22H14C14.5523 22 15 22.4477 15 23V28C15 28.5523 14.5523 29 14 29H12C11.4477 29 11 28.5523 11 28Z" fill="white" fill-opacity="0.9"/>
                        <path d="M17 28V18C17 17.4477 17.4477 17 18 17H20C20.5523 17 21 17.4477 21 18V28C21 28.5523 20.5523 29 20 29H18C17.4477 29 17 28.5523 17 28Z" fill="white" fill-opacity="0.95"/>
                        <path d="M23 28V14C23 13.4477 23.4477 13 24 13H26C26.5523 13 27 13.4477 27 14V28C27 28.5523 26.5523 29 26 29H24C23.4477 29 23 28.5523 23 28Z" fill="white"/>
                        <path d="M29 28V20C29 19.4477 29.4477 19 30 19H32C32.5523 19 33 19.4477 33 20V28C33 28.5523 32.5523 29 32 29H30C29.4477 29 29 28.5523 29 28Z" fill="white" fill-opacity="0.9"/>
                        <path d="M12 20L19 14L25 11L32 16" stroke="#A5F3FC" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
                        <circle cx="25" cy="11" r="2.5" fill="#FFFFFF"/>
                    </svg>
                </div>
                <h2 style="color: #0F172A; font-weight: 800; margin-bottom: 4px; letter-spacing: -0.02em;">
                    Customer Review Intelligence
                </h2>
                <div style="font-size: 0.95rem; color: #475569; font-weight: 500;">
                    Enterprise Access Portal · Cloud Firestore Synced
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_register, tab_evaluator = st.tabs([
            "🔑 Sign In",
            "📝 Create Account (Sign Up)",
            "🎓 Quick Demo Access",
        ])

        # ---------------------------------------------------------
        # TAB 1: SIGN IN
        # ---------------------------------------------------------
        with tab_login:
            st.markdown("##### Sign in to access your intelligence workspace")
            login_email = st.text_input(
                "Email Address",
                key="login_email_input",
                placeholder="analyst@enterprise.com",
            )
            login_password = st.text_input(
                "Password",
                type="password",
                key="login_password_input",
                placeholder="Enter your password",
            )

            col_btn, col_hint = st.columns([1, 1])
            with col_btn:
                btn_signin = st.button("🚀 Sign In", use_container_width=True, type="primary")

            if btn_signin:
                if not login_email or not login_password:
                    st.error("Please enter both your email address and password.")
                else:
                    is_ok, msg, user_data = auth_mgr.login_user(login_email, login_password)
                    if is_ok and user_data:
                        st.session_state["authenticated"] = True
                        st.session_state["current_user"] = user_data
                        st.success(f"Welcome back, {user_data.get('name', 'Analyst')}!")
                        st.rerun()
                    else:
                        st.error(msg)

        # ---------------------------------------------------------
        # TAB 2: CREATE ACCOUNT (WITH STRICT CONSTRAINTS)
        # ---------------------------------------------------------
        with tab_register:
            st.markdown("##### Create a new authenticated profile")

            reg_name = st.text_input(
                "Full Name",
                key="reg_name_input",
                placeholder="e.g. Alex Rivera",
            )
            reg_email = st.text_input(
                "Email Address (must contain '@')",
                key="reg_email_input",
                placeholder="user@enterprise.com",
            )
            reg_role = st.selectbox(
                "Workspace Role",
                options=[
                    "Product Intelligence Analyst",
                    "Senior NLP Engineer",
                    "Customer Experience Director",
                    "Academic Evaluator / Professor",
                ],
                key="reg_role_select",
            )

            reg_password = st.text_input(
                "Password (min 8 chars, uppercase, lowercase, digit, symbol)",
                type="password",
                key="reg_pass_input",
                placeholder="e.g. StrongPass2026!",
            )
            reg_confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                key="reg_confirm_pass_input",
                placeholder="Re-type password",
            )

            # Live constraint feedback box
            p_len = len(reg_password) >= 8
            p_upper = any(c.isupper() for c in reg_password)
            p_lower = any(c.islower() for c in reg_password)
            p_digit = any(c.isdigit() for c in reg_password)
            p_sym = any(c in r"""!@#$%^&*(),.?":{}|<>\-_+=[]~`/\\""" for c in reg_password)
            p_match = (reg_password == reg_confirm_password) if reg_password else False

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.caption(f"{'✅' if p_len else '⚪'} Min 8 characters")
                st.caption(f"{'✅' if p_upper and p_lower else '⚪'} Upper & lowercase")
            with col_c2:
                st.caption(f"{'✅' if p_digit and p_sym else '⚪'} Number & special symbol")
                st.caption(f"{'✅' if p_match else '⚪'} Passwords match")

            btn_register = st.button("✨ Create Account & Sync to Firebase", use_container_width=True, type="primary")

            if btn_register:
                is_ok, msg, user_data = auth_mgr.register_user(
                    email=reg_email,
                    password=reg_password,
                    confirm_password=reg_confirm_password,
                    name=reg_name,
                    role=reg_role,
                )
                if is_ok and user_data:
                    st.session_state["authenticated"] = True
                    st.session_state["current_user"] = user_data
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        # ---------------------------------------------------------
        # TAB 3: QUICK DEMO ACCESS FOR EVALUATORS / VIVA
        # ---------------------------------------------------------
        with tab_evaluator:
            st.info(
                "💡 **Academic Presentation Quick Access:**\n\n"
                "Evaluating or demonstrating this project for academic review? "
                "Click below to immediately log in as a pre-authenticated **Lead Reviewer** without having to register a new account."
            )
            if st.button("⚡ Instant Evaluator Demo Access", use_container_width=True):
                guest_user = auth_mgr.get_guest_user()
                st.session_state["authenticated"] = True
                st.session_state["current_user"] = guest_user
                st.success("Signed in as Academic Evaluator!")
                st.rerun()

    return False


def render_user_sidebar(user_profile: Optional[Dict[str, Any]] = None):
    """Renders active user session card and sign-out control in the sidebar."""
    if not user_profile:
        return

    name = user_profile.get("name", "Analyst")
    email = user_profile.get("email", "user@enterprise.com")
    role = user_profile.get("role", "Analyst")
    provider = user_profile.get("auth_provider", "Verified Session")

    st.sidebar.markdown(
        f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px; margin-bottom:14px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="font-size:1.5rem;">👤</div>
                <div>
                    <div style="font-size:0.92rem; font-weight:700; color:#0F172A; line-height:1.2;">{name}</div>
                    <div style="font-size:0.75rem; color:#64748B;">{email}</div>
                </div>
            </div>
            <div style="margin-top:8px; display:flex; gap:6px; flex-wrap:wrap;">
                <span style="background:#EEF2FF; border:1px solid #C7D2FE; color:#3730A3; padding:2px 8px; border-radius:4px; font-size:0.7rem; font-weight:600;">{role}</span>
                <span style="background:#F1F5F9; border:1px solid #CBD5E1; color:#475569; padding:2px 8px; border-radius:4px; font-size:0.7rem;">{provider}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("🚪 Sign Out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["current_user"] = None
        st.rerun()
