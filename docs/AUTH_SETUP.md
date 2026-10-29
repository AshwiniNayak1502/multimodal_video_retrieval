# Authentication Setup & Configuration Guide

This guide walks through configuring **Gmail Email Verification (SMTP)** and **"Continue with Google" (OpenID Connect)** for VideoContextEngine.

---

## 📋 Overview of Changes & Network Requirements

By default, VideoContextEngine operates 100% offline using Argon2id password hashing and local SQLite storage. When email verification or Google sign-in are configured and enabled:
- **Email Verification:** Sends outbound SMTP requests to `smtp.gmail.com` (port 587 or 465).
- **Google Sign-In:** Performs server-side OIDC authorization redirects and token verification with Google OAuth servers (`accounts.google.com`).

---

## 🔑 Step 1: Generate Cryptographic Secrets

Generate two strong, random 48-byte secrets using Python in PowerShell:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Run this command twice:
1. First value: Use for `VRS_AUTH_SECRET` in `.env` (used for keyed HMAC-SHA256 verification code digests).
2. Second value: Use for `cookie_secret` in `.streamlit/secrets.toml` (used for Streamlit OIDC session cookie signing).

---

## ✉️ Step 2: Configure Dedicated Gmail SMTP Sender

To send email verification codes via Gmail, use a **dedicated** Google account (do **not** use your personal Google account).

1. **Enable 2-Step Verification:**
   - Log into your dedicated Google account at [myaccount.google.com](https://myaccount.google.com/).
   - Navigate to **Security** -> **How you sign in to Google** -> **2-Step Verification** and turn it ON.
2. **Generate an App Password:**
   - In **2-Step Verification**, scroll to the bottom and select **App passwords** (or search "App passwords" in the top search bar).
   - *Note:* Google periodically updates its account navigation; refer to Google Account Help if menu names change.
   - Enter an app name (e.g. `VideoContextEngine`) and click **Create**.
   - Google displays a 16-character code (e.g. `abcd efgh ijkl mnop`). Copy this code.
3. **Configure `.env`:**
   ```ini
   VRS_REQUIRE_EMAIL_VERIFICATION=true
   VRS_AUTH_SECRET="<your-generated-48-byte-secret-from-step-1>"
   VRS_SMTP_HOST=smtp.gmail.com
   VRS_SMTP_PORT=587
   VRS_SMTP_SECURITY=starttls
   VRS_SMTP_USERNAME=your-dedicated-account@gmail.com
   VRS_SMTP_PASSWORD=abcdefghijklmnop
   VRS_EMAIL_FROM=your-dedicated-account@gmail.com
   VRS_EMAIL_FROM_NAME="VideoContextEngine"
   ```

---

## 🌐 Step 3: Configure Google Cloud Console for "Continue with Google"

1. **Open Google Cloud Console:**
   - Visit [console.cloud.google.com](https://console.cloud.google.com/) and create a new project (e.g., `VideoContextEngine-Auth`).
2. **Configure OAuth Consent Screen:**
   - Navigate to **APIs & Services** -> **OAuth consent screen** (or **Google Auth Platform** -> **Branding**).
   - User Type: Select **External** and click **Create**.
   - Fill in required app details:
     - **App name:** `VideoContextEngine`
     - **User support email:** your email
     - **Developer contact information:** your email
   - Click **Save and Continue**.
   - Under **Scopes**, no sensitive or restricted scopes are required (standard `openid`, `profile`, `email` are requested automatically).
3. **Add Test Users (Mandatory in Testing Status):**
   - Under **Test users**, click **+ ADD USERS**.
   - Add every Google account email address that will test or sign into the application.
   - *Important:* While the app is in "Testing" mode, only accounts explicitly added here can authenticate.
4. **Create OAuth 2.0 Client Credentials:**
   - Navigate to **APIs & Services** -> **Credentials** -> **+ CREATE CREDENTIALS** -> **OAuth client ID**.
   - **Application type:** Select **Web application**.
   - **Name:** `VideoContextEngine Web Client`.
   - **Authorized JavaScript origins:** Leave blank (not required for server-side redirects).
   - **Authorized redirect URIs:** Click **+ ADD URI** and add:
     ```
     http://localhost:8501/oauth2callback
     ```
     *Notice:* Must match your Streamlit running port character-for-character.
   - Click **Create**. Copy the **Client ID** and **Client Secret**.

---

## 📁 Step 4: Create and Populate Configuration Files

Copy the configuration templates to their live local paths:

```powershell
# Copy environment configuration
Copy-Item .env.example .env

# Copy Streamlit secrets template
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

1. In `.env`:
   - Set `VRS_ENABLE_GOOGLE=true`.
   - Fill in your `VRS_AUTH_SECRET`, `VRS_SMTP_USERNAME`, and `VRS_SMTP_PASSWORD`.
2. In `.streamlit/secrets.toml`:
   ```toml
   [auth]
   redirect_uri = "http://localhost:8501/oauth2callback"
   cookie_secret = "<your-second-generated-48-byte-secret-from-step-1>"
   client_id = "<your-client-id>.apps.googleusercontent.com"
   client_secret = "<your-client-secret>"
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   expose_tokens = "id"
   ```
3. Restart your Streamlit app:
   ```powershell
   streamlit run app/streamlit_app.py
   ```

---

## 🔍 Troubleshooting & FAQs

### 1. Gmail SMTP "Authentication failed" / `SMTPAuthenticationError (535)`
- **Cause:** Using standard Google account password instead of a Google App Password, or 2-Step Verification is turned off.
- **Fix:** Turn on 2-Step Verification and generate a 16-character App Password at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords).

### 2. `redirect_uri_mismatch` (400 Error on Google Consent Screen)
- **Cause:** The redirect URI in Google Cloud Console does not match `http://localhost:8501/oauth2callback`.
- **Fix:** Ensure the protocol is `http://`, the port matches your running Streamlit port (default 8501), and the path is exactly `/oauth2callback`.

### 3. "Access blocked: app has not completed the Google verification process" / Error 403
- **Cause:** The Google account attempting to sign in is not registered under **Test users** in the Google Cloud Console.
- **Fix:** Go to Google Cloud Console -> **OAuth consent screen** -> **Test users** -> **Add users**, and add your email.

### 4. Port Mismatch (e.g. running on 8502)
- **Cause:** If another process is using port 8501, Streamlit defaults to port 8502.
- **Fix:** Either kill the process on 8501 or update both `.streamlit/secrets.toml` and Google Cloud Console authorized redirect URIs to port 8502.

### 5. Gmail Daily Sending Quotas
- **Consumer Gmail:** Capped at 500 emails/24-hour rolling window.
- **Google Workspace:** Capped at 2,000 emails/24-hour rolling window.
- VideoContextEngine includes built-in rate limiters (`VRS_RESEND_COOLDOWN_SECONDS=60`, `VRS_RESEND_MAX_PER_HOUR=5`, `VRS_EMAIL_MAX_PER_HOUR=50`) to protect against quota exhaustion.

---

## ✅ User Manual Verification Checklist

Once credentials are configured, execute these manual tests in your browser:

- [ ] **1. Fresh Registration:** Register with a real inbox. Verify the 6-digit verification code arrives and entering it logs you in.
- [ ] **2. Wrong Code:** Enter an invalid code. Confirm "Invalid verification code. Please try again." appears and increments attempt count.
- [ ] **3. Resend Code:** Click "Resend Code" after the 60-second cooldown countdown expires. Confirm a new code arrives and the previous code is rejected.
- [ ] **4. Change Email:** Click "Change Email" on the verification screen. Confirm the pending state is deleted and the form opens with your name and email prefilled.
- [ ] **5. Continue with Google:** Click "Continue with Google" on Log In tab. Authenticate through Google consent screen. Confirm account is created and logged in.
- [ ] **6. Cancelled Consent:** Click "Continue with Google" and click "Cancel" on Google's consent screen. Verify you return cleanly to the login screen with no crash.
- [ ] **7. Account Linking:** Register an account via email/password. Then sign in with the same email via Google. Confirm accounts are safely linked.
- [ ] **8. Log Out:** Log out from the sidebar. Verify you are logged out and refreshing the browser does NOT silently log you back into Google.
- [ ] **9. Pre-existing Legacy Account:** Log in with an unverified account created before email verification was enabled. Verify you are prompted for a verification code before receiving a session.

