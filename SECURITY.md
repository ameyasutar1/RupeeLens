# Security and Privacy

This application processes private financial statements locally.

## Never commit

- `.env` or API keys
- bank statement CSV files
- SQLite databases
- trained model artifacts
- application logs or exported chat histories

These files are excluded by `.gitignore`.

When deployed on Vercel, use a **private** Vercel Blob store. Never make the SQLite
snapshot public, because it contains financial transactions and chat history.

Authentication uses scrypt password hashes and random revocable session tokens. The browser
receives only an `HttpOnly`, `SameSite=Strict` cookie; the database stores a SHA-256 hash of
the token rather than the bearer token itself. Login failures are rate-limited.

Each account has a separate SQLite database, XGBoost artifact directory, Artha conversation
history, learned rules, and Vercel Blob snapshot. Do not replace this isolation with a
client-supplied user identifier.

On Vercel, set `RUPEELENS_SIGNUP_CODE` before enabling account creation. Keep the Blob store
private. `RUPEELENS_USERNAME` and `RUPEELENS_PASSWORD` may bootstrap the first account;
the password must contain at least 12 characters.

## API keys

Create `.env` from `.env.example`. If a key is ever committed, pasted into an issue,
or otherwise exposed, revoke it immediately and create a replacement.

## Database access

The AI agent receives read-only SQL access. Database mutations are restricted to
application tools that create pending actions and require explicit user approval.
