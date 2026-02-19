# Repository Guidelines

## Project Structure & Module Organization
- `streamlit_app.py`: main Streamlit entrypoint for the unified dashboard.
- `app_sections/`: reusable UI/feature sections consumed by `streamlit_app.py`.
- `apps/`: standalone tools (`content_analyzer`, `gsc_insights`, `linking_optimizer`, `fanout_analyzer`, `brief_generator`) with local `app.py` and feature modules.
- `shared/`: cross-app services (config, auth/licensing, GSC, data orchestration, UI helpers).
- `api/`: FastAPI service (`api/app.py`) for project and analysis endpoints.
- `scripts/`: deployment and server automation (`deploy.sh`, `setup-server.sh`, `ssl-setup.sh`).
- `workspace/`: runtime project data and exports; treat as environment state, not source code.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create/activate local environment.
- `pip install -r requirements.txt`: install full dashboard dependencies.
- `streamlit run streamlit_app.py`: run the main dashboard locally.
- `streamlit run apps/content_analyzer/app.py`: run one standalone app (swap path for other apps).
- `python verify_setup.py`: environment and NLP dependency smoke check.
- `uvicorn api.app:app --host 0.0.0.0 --port 8001 --reload`: run the FastAPI service.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, constants in `UPPER_SNAKE_CASE`.
- Keep imports grouped: standard library, third-party, local modules.
- Prefer small, composable functions in `shared/` and `apps/*/modules/` over large monolithic handlers.
- Follow existing bilingual context (English/Spanish) but keep identifiers consistent and descriptive.

## Testing Guidelines
- Current repository practice is smoke/integration-oriented, not strict unit-test coverage.
- Before opening a PR, run `python verify_setup.py` and manually validate changed flows in Streamlit.
- If you add automated tests, use `test_<feature>.py` naming and include run instructions in the PR.

## Commit & Pull Request Guidelines
- Follow Conventional Commit prefixes seen in history: `feat:`, `fix:`, `chore:`, `security:`.
- Keep commits focused and scoped to one change area (e.g., `fix: normalize embedding parsing in API upload`).
- PRs should include: summary, affected paths, manual test steps, env/config changes, and screenshots for UI updates.

## Security & Configuration Tips
- Never commit secrets; use `.env` from `.env.example` and Streamlit secrets locally.
- Do not persist user API keys in code or saved session artifacts.
- Treat `workspace/projects/` data as sensitive; sanitize examples before sharing.
