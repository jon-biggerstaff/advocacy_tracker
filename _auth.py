"""
Form-based auth for the Dash app, backed by Flask session cookies.

Call `init_auth(server)` with the Flask server (i.e. `app.server` on a Dash
app). If either APP_USERNAME or APP_PASSWORD env vars are missing, auth is
disabled entirely — this keeps local dev friction-free. On Render, set both
env vars and a strong SECRET_KEY and every request is gated behind /login
until the user has a valid session cookie.

Notes
- Sessions are Flask's default signed-cookie sessions, so no server-side
  session store is required. Cookies expire after `SESSION_DAYS`.
- Passwords are compared with `hmac.compare_digest` to avoid timing leaks.
- The login page is rendered as a bare HTML template — it does not depend
  on Dash so it renders before any of the Dash bundle loads.
"""
from __future__ import annotations

import hmac
import os
import secrets
from datetime import timedelta
from urllib.parse import urlparse, urljoin

from flask import (
    Flask,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)
from werkzeug.middleware.proxy_fix import ProxyFix

SESSION_DAYS = 30

# ── HTML template ─────────────────────────────────────────────────────────
# Standalone HTML so the login screen renders instantly without pulling in
# the Dash JS bundle. Styling matches the dashboard aesthetic.
_LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Sign in — Advocacy Fund Tracker</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: #f0f0ec;
            color: #292524;
            font-family: 'DM Sans', sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .card {
            background: #ffffff;
            border: 1px solid #e2e2dc;
            border-radius: 12px;
            padding: 32px;
            width: 100%;
            max-width: 380px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        }
        .brand {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 28px;
            letter-spacing: 2px;
            color: #292524;
            margin-bottom: 4px;
            line-height: 1;
        }
        .brand-sub {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            letter-spacing: 2px;
            color: #57534e;
            margin-bottom: 24px;
            text-transform: uppercase;
        }
        label {
            display: block;
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            letter-spacing: 2px;
            color: #57534e;
            margin: 16px 0 6px 0;
            text-transform: uppercase;
        }
        input[type="text"], input[type="password"] {
            width: 100%;
            height: 40px;
            font-family: 'DM Mono', monospace;
            font-size: 14px;
            color: #292524;
            background: #ffffff;
            border: 1px solid #e2e2dc;
            border-radius: 8px;
            padding: 0 12px;
            outline: none;
            transition: border-color 0.15s ease;
        }
        input[type="text"]:focus, input[type="password"]:focus {
            border-color: #0369a1;
        }
        button {
            width: 100%;
            height: 42px;
            margin-top: 24px;
            font-family: 'DM Mono', monospace;
            font-size: 12px;
            letter-spacing: 2px;
            color: #ffffff;
            background: #0369a1;
            border: 0;
            border-radius: 8px;
            cursor: pointer;
            text-transform: uppercase;
            transition: background 0.15s ease;
        }
        button:hover { background: #075985; }
        .error {
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            letter-spacing: 1px;
            color: #dc2626;
            background: rgba(220,38,38,0.08);
            border: 1px solid rgba(220,38,38,0.2);
            border-radius: 6px;
            padding: 8px 12px;
            margin-top: 16px;
            text-transform: uppercase;
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="brand">Fund Tracker</div>
        <div class="brand-sub">Sign in to continue</div>
        <form method="post" action="{{ post_url }}">
            <label for="username">Username</label>
            <input type="text" name="username" id="username" autocomplete="username" required autofocus />
            <label for="password">Password</label>
            <input type="password" name="password" id="password" autocomplete="current-password" required />
            <button type="submit">Sign in</button>
            {% if error %}<div class="error">{{ error }}</div>{% endif %}
        </form>
    </div>
</body>
</html>
"""


def _is_safe_next(target: str | None) -> bool:
    """Only allow redirects to same-host relative URLs."""
    if not target:
        return False
    ref_url  = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ("http", "https") and ref_url.netloc == test_url.netloc


def init_auth(server: Flask) -> None:
    """
    Register /login + /logout on the Flask server and gate every other
    request behind a valid session. No-op if credentials aren't configured.
    """
    username = os.environ.get("APP_USERNAME")
    password = os.environ.get("APP_PASSWORD")

    if not (username and password):
        server.logger.warning(
            "APP_USERNAME / APP_PASSWORD not set — auth is DISABLED. "
            "Set both env vars (plus SECRET_KEY) to enable login."
        )
        return

    # Secret key: prefer env var; fall back to an ephemeral random one so
    # cookies still work in a single-process dev run (they'll be invalidated
    # on every restart, which is fine).
    server.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    server.permanent_session_lifetime = timedelta(days=SESSION_DAYS)

    # When deployed on Render (auto-detected via the RENDER env var, which
    # Render always sets), tell Flask to trust the X-Forwarded-Proto header
    # from Render's edge proxy AND set the session cookie flags required
    # for the cookie to survive being embedded in a cross-site iframe like
    # Notion. Without this, Chrome silently drops the session cookie on the
    # post-login redirect and the user gets bounced back to /login in a
    # loop — the exact "login form takes creds but nothing happens" symptom
    # seen when the dashboard is embedded in Notion.
    if os.environ.get("RENDER") or os.environ.get("RENDER_EXTERNAL_HOSTNAME"):
        server.wsgi_app = ProxyFix(server.wsgi_app, x_for=1, x_proto=1, x_host=1)
        server.config.update(
            SESSION_COOKIE_SAMESITE="None",  # allow cross-site iframe use
            SESSION_COOKIE_SECURE=True,      # required when SameSite=None
            SESSION_COOKIE_HTTPONLY=True,    # no JS access to the cookie
        )

    @server.route("/login", methods=["GET", "POST"])
    def _login():
        error = None
        if request.method == "POST":
            submitted_user = (request.form.get("username") or "").strip()
            submitted_pass = request.form.get("password") or ""
            user_ok = hmac.compare_digest(submitted_user.encode(), username.encode())
            pass_ok = hmac.compare_digest(submitted_pass.encode(), password.encode())
            if user_ok and pass_ok:
                session.permanent = True
                session["auth_user"] = submitted_user
                next_url = request.args.get("next") or request.form.get("next") or "/"
                return redirect(next_url if _is_safe_next(next_url) else "/")
            error = "Invalid username or password"

        next_url = request.args.get("next") or "/"
        post_url = url_for("_login") + (f"?next={next_url}" if next_url != "/" else "")
        return render_template_string(_LOGIN_HTML, error=error, post_url=post_url)

    @server.route("/logout", methods=["GET", "POST"])
    def _logout():
        session.clear()
        return redirect(url_for("_login"))

    @server.before_request
    def _require_login():
        # Endpoints and asset prefixes that should always be reachable.
        allowed_endpoints = {"_login", "_logout"}
        allowed_prefixes  = ("/assets/", "/favicon.ico")
        if request.endpoint in allowed_endpoints:
            return None
        if any(request.path.startswith(p) for p in allowed_prefixes):
            return None
        if session.get("auth_user"):
            return None
        # Preserve where they were headed so we can bounce them back after login.
        return redirect(url_for("_login", next=request.full_path if request.query_string else request.path))
