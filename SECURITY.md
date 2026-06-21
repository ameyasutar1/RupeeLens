# Security and Privacy

This application processes private financial statements locally.

## Never commit

- `.env` or API keys
- bank statement CSV files
- SQLite databases
- trained model artifacts
- application logs or exported chat histories

These files are excluded by `.gitignore`.

## API keys

Create `.env` from `.env.example`. If a key is ever committed, pasted into an issue,
or otherwise exposed, revoke it immediately and create a replacement.

## Database access

The AI agent receives read-only SQL access. Database mutations are restricted to
application tools that create pending actions and require explicit user approval.
