# Lyre Tutor — Render deployment

## Files
- `app.py`: your existing FastAPI/Lyre Tutor API
- `requirements.txt`: Python dependencies
- `.python-version`: requests Python 3.11
- `render.yaml`: Render Blueprint configuration

## Deploy
1. Upload these files to a GitHub repository.
2. In Render, choose **New → Blueprint** and connect that repository.
3. When prompted, set `GEMINI_API_KEY` to your Gemini API key.
4. Set `API_SECRET` to a long random secret. Use the exact same value in the n8n HTTP Request nodes' `X-API-Key` header.
5. Deploy. Your base URL will look like `https://lyre-tutor-api.onrender.com` (Render may assign a different service name).
6. Test `GET https://YOUR-SERVICE.onrender.com/health`; expected response: `{"status":"ok"}`.

## n8n changes
Keep the existing Telegram Trigger, branching, and Telegram send-message nodes.
- Text branch: `POST https://YOUR-SERVICE.onrender.com/agent`
  - Header: `X-API-Key: <same API_SECRET>`
  - JSON body: `chat_id` and `text`
- Audio branch: `POST https://YOUR-SERVICE.onrender.com/listen`
  - Header: `X-API-Key: <same API_SECRET>`
  - Body type: multipart/form-data
  - Fields: `chat_id` (text) and `file` (binary file)
  - In n8n, make sure the binary property name used by the HTTP Request node matches the output of the Telegram Get a file node.

## Important limitations
- Render Free services can sleep when idle and may take time to wake up.
- The app currently keeps conversation checkpoints, expected song notes, and pending recordings in process memory. A restart/redeploy can reset conversation state. For durable memory across restarts, replace the in-memory checkpoint and dictionaries with persistent storage (for example, a database).
- Render's temporary filesystem is fine for temporary audio uploads, but do not store permanent user data there.
- Code updates pushed to the connected GitHub branch can trigger automatic redeployment. Editing a Colab notebook alone does not deploy changes: the updated Python source must first be synced to GitHub.
- Dependencies are intentionally not pinned yet. After a successful first deployment, pin tested versions to avoid unexpected dependency changes.
