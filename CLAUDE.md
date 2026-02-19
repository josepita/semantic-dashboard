# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Embedding Insights Dashboard Suite** - A collection of SEO and content analysis tools built with Streamlit, utilizing semantic embeddings, NLP, and AI APIs to analyze content, keywords, and search performance.

### Core Technologies
- **Streamlit**: All UI applications
- **Python 3.10+**: Backend
- **Sentence Transformers**: Semantic embeddings
- **OpenAI API**: Embeddings and GPT models
- **Google APIs**: Gemini AI, Search Console, OAuth
- **spaCy**: NLP and named entity recognition
- **DuckDB**: Local database for project data
- **Pandas/Polars**: Data processing

## Architecture

### Directory Structure

```
/
├── streamlit_app.py           # Main hub dashboard
├── app_sections/              # UI sections for main dashboard
├── apps/                      # Standalone applications
│   ├── brief_generator/       # SEO brief generator with SERP analysis
│   ├── content_analyzer/      # Content semantic analysis tool
│   ├── fanout_analyzer/       # Query fan-out analysis
│   ├── gsc_insights/          # Google Search Console insights
│   └── linking_optimizer/     # Internal linking optimization
├── shared/                    # Shared utilities across all apps
│   ├── project_manager.py     # Multi-project workspace management
│   ├── oauth_manager.py       # Google OAuth handling
│   ├── license_manager.py     # License verification system
│   ├── gsc_client.py          # Google Search Console client
│   ├── gemini_utils.py        # Gemini AI utilities
│   └── ...                    # Other shared modules
├── workspace/                 # User data and projects (gitignored)
└── .env                       # API keys and config (gitignored)
```

### Application Architecture

- **Main Dashboard** (`streamlit_app.py`): Central hub with multiple analysis sections
- **Standalone Apps**: Each app in `apps/` is self-contained with own `app.py` and `modules/`
- **Shared Layer**: Common functionality (auth, project management, API clients) in `shared/`
- **Path Resolution**: Apps use relative imports and add `shared/` to `sys.path` for cross-references
- **Project System**: Multi-project workspace in `workspace/projects/{project_name}/`

### Key Systems

1. **Project Manager** (`shared/project_manager.py`)
   - Creates isolated workspaces per project
   - Manages DuckDB databases, OAuth credentials, embeddings cache
   - Structure: `workspace/projects/{project}/[data/oauth/cache/]`

2. **License Manager** (`shared/license_manager.py`)
   - Feature gating based on license tier
   - Currently in dev mode (`LICENSE_DEV_MODE = True` in `shared/config.py`)
   - Server-based validation with local 24h cache

3. **OAuth Manager** (`shared/oauth_manager.py`)
   - Google OAuth flow for Search Console and Analytics
   - Per-project credential storage in `oauth/` directory
   - Encryption of stored tokens

## Development Commands

### Setup (Windows)
```bash
# Install dependencies and create virtual environment
INSTALAR.bat

# Run main dashboard
EJECUTAR.bat

# Or manually:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download es_core_news_md
streamlit run streamlit_app.py
```

### Setup (Linux/Mac)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_md  # or en_core_web_sm
streamlit run streamlit_app.py
```

### Running Individual Apps
```bash
# Each app can run standalone
cd apps/content_analyzer
streamlit run app.py

# Or use launcher scripts (Windows):
# - "Launcher - Brief Generator.bat"
# - "Launcher - Content Analyzer.vbs"
# - "Launcher - GSC Insights.vbs"
# etc.
```

### Environment Configuration
```bash
# Copy template and fill in API keys
cp .env.example .env

# Required API keys:
# - OPENAI_API_KEY (for embeddings)
# - GEMINI_API_KEY (for AI analysis)
# - SERPROBOT_API_KEY (for SERP data in Brief Generator)
```

### Testing
```bash
# Validate setup
python verify_setup.py

# Run system tests (CI/CD)
python test_system.py
```

## Common Development Patterns

### Adding a New App Section
1. Create module in `app_sections/` with `render_<section_name>()` function
2. Import in `streamlit_app.py`
3. Add navigation option in `render_sidebar_navigation()`
4. Add view handling in `main()` function

### Adding a New Standalone App
1. Create directory in `apps/<app_name>/`
2. Add `app.py` with path setup to import from `shared/`
3. Create `modules/` subdirectory for app-specific logic
4. Add launcher script (`.bat` or `.vbs`) in root directory

### Working with Shared Utilities
- Import from `shared/` using: `from shared.module_name import function`
- For apps: Add `sys.path.insert(0, str(shared_path))` before imports
- Common imports:
  - `from shared.project_manager import ProjectManager`
  - `from shared.gsc_client import GSCClient`
  - `from shared.gemini_utils import call_gemini`

### Session State Management
- Main dashboard uses unprefixed session state: `st.session_state["processed_df"]`
- Standalone apps use prefixed keys: `st.session_state[f"{SS}keyword"]` where `SS = "app_prefix_"`
- Project context stored in: `st.session_state["current_project"]`

### Working with Embeddings
- Use `sentence-transformers` for local embeddings
- Default model: `paraphrase-multilingual-MiniLM-L12-v2`
- Embeddings cached per project in `workspace/projects/{project}/cache/`
- Utility: `shared/embedding_cache.py`

## Important Notes

### API Key Security
- **NEVER commit API keys**: All keys go in `.env` (gitignored)
- Apps prompt for keys in UI if not in environment
- Per-project API keys can be stored in `workspace/projects/{project}/oauth/`

### License System
- Currently disabled: `LICENSE_DEV_MODE = True` in `shared/config.py`
- When enabled: Features gated by `require_feature()` in `shared/license_ui.py`
- License check happens at app startup via `init_license_check()`

### spaCy Models
- Spanish: `es_core_news_sm` or `es_core_news_md`
- English: `en_core_web_sm`
- Download before first run: `python -m spacy download <model_name>`
- Used for NER, lemmatization, and entity extraction

### Google Search Console Integration
- Requires OAuth: Apps guide through flow in UI
- Credentials stored per-project: `workspace/projects/{project}/oauth/gsc_token.json`
- Client: `shared/gsc_client.py`
- UI helpers: `shared/gsc_ui.py`

### Workspace Data
- All user data in `workspace/` (gitignored)
- Project structure auto-created by `ProjectManager`
- DuckDB databases: `workspace/projects/{project}/data/{app}.db`
- Backups: Can be enabled in project settings

## Deployment

### Docker (Production)
```bash
# Build and deploy
docker compose up -d

# Uses Gunicorn for production serving
# Multi-domain routing configured in Caddy/nginx
```

### CI/CD
- GitHub Actions workflow: `.github/workflows/deploy.yml`
- Runs on push to `main` or `master`
- Steps: Test → Build Docker → Deploy → Health Check
- Automated backups: Weekly on Sundays

### Production Environment Variables
Required in production `.env`:
- `ENVIRONMENT=production`
- `STREAMLIT_SERVER_HEADLESS=true`
- All API keys (OpenAI, Gemini, Serprobot)
- Domain configurations for multi-app deployment
