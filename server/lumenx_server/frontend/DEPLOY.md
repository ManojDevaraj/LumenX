# Hosting the frontend separately

The FastAPI server already serves this frontend at `/`. To host the frontend on its own
(Vercel, Netlify, GitHub Pages) and point it at a backend elsewhere:

1. **Backend:** start the server with the frontend's origin allowed:

   ```bash
   LUMENX_CORS_ORIGINS="https://your-site.vercel.app" \
     core/.venv/bin/python -m uvicorn lumenx_server.main:app --app-dir server --host 0.0.0.0 --port 8000
   ```

   Expose it publicly, for example with `cloudflared tunnel --url http://localhost:8000`.

2. **Frontend:** set the backend URL (no trailing slash) in `config.js`:

   ```js
   window.LUMENX_CONFIG = { apiBase: 'https://your-tunnel.trycloudflare.com' };
   ```

3. **Deploy** this `frontend/` folder as a static site. `vercel.json` maps `/static/*` to the
   folder root so the page's asset paths resolve.

The landing scene and story play back a recorded run from `showcase/`, so they work even if
the backend is asleep. Re-record it with `server/scripts/build_showcase.py` after engine changes.
Gallery photos come from the backend (`server/scripts/fetch_demo_photos.py`).
