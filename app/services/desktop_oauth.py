"""Desktop-native Google OAuth using system default browser, loopback redirect, and PKCE.

Bypasses Google's 403 disallowed_useragent restriction on embedded webviews (Edge WebView2)
by completing authentication in the user's regular browser (Chrome, Edge, Firefox, etc.)
and securely handing off the authenticated session back to VideoContextEngine.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import http.server
import json
import logging
import os
import secrets
import socket
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from ..db.app_db import AppDatabase, get_db
from ..settings import settings
from .auth_service import User, google_sign_in

logger = logging.getLogger("app.oauth")

# In-memory thread-safe flow and handoff registries
_lock = threading.Lock()
_ACTIVE_FLOWS: Dict[str, Dict[str, Any]] = {}
_HANDOFF_TOKENS: Dict[str, Dict[str, Any]] = {}


def get_google_oauth_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Retrieve Google OAuth client_id and client_secret from secrets or environment."""
    client_id = os.getenv("GOOGLE_CLIENT_ID") or os.getenv("VRS_GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET") or os.getenv("VRS_GOOGLE_CLIENT_SECRET")

    if not client_id or not client_secret:
        try:
            from streamlit.runtime.secrets import secrets_singleton
            auth_sec = secrets_singleton.get("auth", {})
            google_sec = auth_sec.get("google", {})
            client_id = client_id or auth_sec.get("client_id") or google_sec.get("client_id")
            client_secret = client_secret or auth_sec.get("client_secret") or google_sec.get("client_secret")
        except Exception:
            pass

    return client_id, client_secret


def is_desktop_google_configured() -> bool:
    """Check if desktop Google sign-in is enabled and credentials are available."""
    is_enabled = (
        os.getenv("VRS_ENABLE_GOOGLE", "").lower() in ("true", "1", "yes")
        if "VRS_ENABLE_GOOGLE" in os.environ
        else settings.enable_google
    )
    if not is_enabled:
        return False

    client_id, client_secret = get_google_oauth_credentials()
    if not client_id or not client_secret:
        return False

    # Check for placeholder values
    if "YOUR_GOOGLE" in client_id or "CHANGE_ME" in client_secret:
        return False

    return True


def _generate_pkce_pair() -> Tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def _find_free_port() -> int:
    """Find an available ephemeral port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """Temporary HTTP handler capturing the Google OAuth authorization code redirect."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard HTTP server access logs to prevent noise."""
        pass

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path != "/oauth2callback":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        query_params = urllib.parse.parse_qs(parsed_url.query)
        code = query_params.get("code", [None])[0]
        state = query_params.get("state", [None])[0]
        error = query_params.get("error", [None])[0]

        flow_data = None
        flow_id = None
        with _lock:
            for fid, f in _ACTIVE_FLOWS.items():
                if f.get("state") == state:
                    flow_data = f
                    flow_id = fid
                    break

        if not flow_data:
            self._render_response(
                success=False,
                title="Invalid OAuth State",
                message="Security verification failed: OAuth state parameter mismatch or flow expired. Please return to VideoContextEngine and try again.",
            )
            return

        if error:
            with _lock:
                flow_data["status"] = "failed"
                flow_data["error"] = f"Google authentication was denied: {error}"
            self._render_response(
                success=False,
                title="Sign-In Cancelled",
                message="Google sign-in was cancelled or denied. You can return to VideoContextEngine.",
            )
            return

        if not code:
            with _lock:
                flow_data["status"] = "failed"
                flow_data["error"] = "Authorization code was not returned by Google."
            self._render_response(
                success=False,
                title="Missing Authorization Code",
                message="No authorization code received from Google. Please try again.",
            )
            return

        # Exchange authorization code for tokens via PKCE
        try:
            token_url = "https://oauth2.googleapis.com/token"
            data = {
                "client_id": flow_data["client_id"],
                "client_secret": flow_data["client_secret"],
                "code": code,
                "code_verifier": flow_data["code_verifier"],
                "grant_type": "authorization_code",
                "redirect_uri": flow_data["redirect_uri"],
            }
            token_resp = requests.post(token_url, data=data, timeout=15)
            if token_resp.status_code != 200:
                err_text = token_resp.text
                logger.error("Token exchange failed: %d - %s", token_resp.status_code, err_text)
                with _lock:
                    flow_data["status"] = "failed"
                    flow_data["error"] = "Failed to exchange authorization code for tokens."
                self._render_response(
                    success=False,
                    title="Token Exchange Error",
                    message="Google rejected the authorization code. Please try again.",
                )
                return

            tokens = token_resp.json()
            id_token_jwt = tokens.get("id_token")

            # Validate ID token using existing auth_service validation logic
            auth_res = google_sign_in(id_token_jwt)
            if not auth_res.success or not auth_res.user or not auth_res.token:
                err_msg = auth_res.error_message or "Account linking failed."
                with _lock:
                    flow_data["status"] = "failed"
                    flow_data["error"] = err_msg
                self._render_response(
                    success=False,
                    title="Verification Failed",
                    message=err_msg,
                )
                return

            # Success: register handoff token
            handoff_token = secrets.token_urlsafe(32)
            with _lock:
                _HANDOFF_TOKENS[handoff_token] = {
                    "user": auth_res.user,
                    "session_token": auth_res.token,
                    "expires_at": time.time() + 60,  # 60 second single-use TTL
                }
                flow_data["status"] = "completed"
                flow_data["session_token"] = auth_res.token
                flow_data["user"] = auth_res.user
                flow_data["handoff_token"] = handoff_token

            self._render_response(
                success=True,
                title="Sign-In Successful",
                message="You are now logged in! You can safely close this browser tab and return to VideoContextEngine.",
                user_name=auth_res.user.display_name,
            )

        except Exception as exc:
            logger.error("Exception in OAuth callback handler: %s", exc, exc_info=True)
            with _lock:
                flow_data["status"] = "failed"
                flow_data["error"] = f"Sign-in error: {exc}"
            self._render_response(
                success=False,
                title="Unexpected Error",
                message=f"An unexpected error occurred during authentication: {exc}",
            )

    def _render_response(
        self,
        success: bool,
        title: str,
        message: str,
        user_name: Optional[str] = None,
    ) -> None:
        """Render a polished, branded dark-mode HTML response page."""
        self.send_response(200 if success else 400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

        status_color = "#10B981" if success else "#EF4444"
        badge_icon = "✓" if success else "✕"
        greeting = f"<p style='color: #A78BFA; font-size: 1.1rem; margin-bottom: 0.5rem;'>Welcome, <strong>{user_name}</strong>!</p>" if user_name else ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - VideoContextEngine</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0B0F19;
            color: #E2E8F0;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }}
        .card {{
            background: #111827;
            border: 1px solid #1F2937;
            border-radius: 16px;
            padding: 40px;
            max-width: 460px;
            width: 100%;
            text-align: center;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.4);
        }}
        .icon-circle {{
            width: 64px;
            height: 64px;
            border-radius: 50%;
            background: rgba({ '16, 185, 129' if success else '239, 68, 68' }, 0.15);
            color: {status_color};
            font-size: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px auto;
            border: 1px solid rgba({ '16, 185, 129' if success else '239, 68, 68' }, 0.3);
        }}
        h1 {{
            font-size: 1.5rem;
            margin: 0 0 12px 0;
            color: #F8FAFC;
        }}
        p {{
            color: #94A3B8;
            line-height: 1.6;
            margin: 0 0 24px 0;
            font-size: 0.95rem;
        }}
        .brand {{
            font-size: 0.85rem;
            color: #64748B;
            border-top: 1px solid #1F2937;
            padding-top: 20px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="card">
        <div class="icon-circle">{badge_icon}</div>
        <h1>{title}</h1>
        {greeting}
        <p>{message}</p>
        <div class="brand">VideoContextEngine Desktop</div>
    </div>
    <script>
        // Auto-close if popup
        if (window.opener) {{
            setTimeout(function() {{ window.close(); }}, 3000);
        }}
    </script>
</body>
</html>"""
        self.wfile.write(html.encode("utf-8"))


def start_desktop_google_auth() -> Dict[str, Any]:
    """Start desktop OAuth flow: launches default browser and loopback listener.
    
    Returns:
        dict: {"success": bool, "flow_id": str, "auth_url": str, "error": str}
    """
    if not is_desktop_google_configured():
        return {
            "success": False,
            "error": "Google sign-in is not configured. Please set client credentials in settings or environment."
        }

    client_id, client_secret = get_google_oauth_credentials()
    code_verifier, code_challenge = _generate_pkce_pair()
    state = secrets.token_urlsafe(32)
    flow_id = secrets.token_hex(16)
    port = _find_free_port()
    redirect_uri = f"http://127.0.0.1:{port}/oauth2callback"

    flow_data = {
        "flow_id": flow_id,
        "state": state,
        "code_verifier": code_verifier,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "port": port,
        "status": "pending",
        "session_token": None,
        "user": None,
        "error": None,
        "created_at": time.time(),
        "expires_at": time.time() + 300,  # 5 minute timeout
    }

    with _lock:
        _ACTIVE_FLOWS[flow_id] = flow_data

    # Start loopback HTTP server in background thread
    server = http.server.HTTPServer(("127.0.0.1", port), _OAuthCallbackHandler)
    server.timeout = 1.0

    def _serve_until_done():
        deadline = time.time() + 300
        while time.time() < deadline:
            with _lock:
                st = flow_data.get("status")
            if st in ("completed", "failed"):
                break
            server.handle_request()
        server.server_close()

    thread = threading.Thread(target=_serve_until_done, daemon=True, name=f"OAuthServer-{port}")
    thread.start()

    # Construct Google OAuth Authorization URL
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)

    # Launch default web browser
    try:
        webbrowser.open(auth_url)
    except Exception as exc:
        logger.warning("Could not launch default web browser: %s", exc)

    return {
        "success": True,
        "flow_id": flow_id,
        "auth_url": auth_url,
    }


def check_flow_status(flow_id: str) -> Dict[str, Any]:
    """Check the status of an in-progress OAuth flow."""
    with _lock:
        flow = _ACTIVE_FLOWS.get(flow_id)
        if not flow:
            return {"status": "expired", "error": "Authentication flow expired or invalid."}

        if time.time() > flow.get("expires_at", 0):
            flow["status"] = "failed"
            flow["error"] = "Authentication timed out. Please try again."

        return {
            "status": flow.get("status", "pending"),
            "session_token": flow.get("session_token"),
            "user": flow.get("user"),
            "error": flow.get("error"),
        }


def cancel_flow(flow_id: str) -> None:
    """Cancel an active OAuth flow."""
    with _lock:
        if flow_id in _ACTIVE_FLOWS:
            _ACTIVE_FLOWS[flow_id]["status"] = "failed"
            _ACTIVE_FLOWS[flow_id]["error"] = "Sign-in cancelled by user."


def consume_handoff_token(token: str) -> Tuple[Optional[User], Optional[str]]:
    """Single-use redemption of a desktop auth handoff token. Returns (user, session_token)."""
    with _lock:
        entry = _HANDOFF_TOKENS.pop(token, None)
        if not entry:
            return None, None
        if time.time() > entry["expires_at"]:
            return None, None
        return entry["user"], entry["session_token"]

