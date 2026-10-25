"""Main Streamlit Entrypoint for VideoContextEngine."""

from pathlib import Path
import streamlit as st

from app.db.app_db import AppDatabase, get_db
from app.services.auth_service import validate_session
from app.services.ingestion_runner import IngestionRunner, recover_interrupted_jobs
from app.services.search_service import SearchEngineWrapper, get_search_engine
from app.settings import settings
from app.ui.auth_view import render_auth_view
from app.ui.library_view import render_library_view
from app.ui.search_view import render_search_view
from app.ui.shell import render_sidebar
from app.ui.upload_view import render_upload_view

# Streamlit Page Configuration
st.set_page_config(
    page_title="VideoContextEngine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_theme():
    """Inject custom CSS theme into Streamlit app."""
    theme_file = Path(__file__).parent / "ui" / "theme.css"
    if theme_file.is_file():
        css = theme_file.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# Application Resource Singletons
@st.cache_resource
def get_app_database() -> AppDatabase:
    import os
    env_path = os.getenv("VRS_APP_DB_PATH")
    target_path = Path(env_path) if env_path else settings.app_db_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return get_db(target_path)


@st.cache_resource
def get_search_wrapper() -> SearchEngineWrapper:
    return get_search_engine()


@st.cache_resource
def get_ingestion_worker() -> IngestionRunner:
    db = get_app_database()
    # Recover any crashed/interrupted jobs from previous runs
    recover_interrupted_jobs(db)
    
    search_wrapper = get_search_wrapper()
    runner = IngestionRunner(
        db=db,
        on_index_updated=search_wrapper.reload,
    )
    runner.start()
    return runner


@st.fragment(run_every=4)
def poll_pipeline_status():
    """Fragment to poll background ingestion jobs without disrupting user interactions."""
    runner = get_ingestion_worker()
    active_count = runner.active_job_count()
    if active_count > 0:
        st.caption(f"⚙️ Ingestion pipeline active: {active_count} video(s) in queue/processing.")


def main():
    load_theme()

    database = get_app_database()

    # Session authentication validation on every rerun
    session_token = st.session_state.get("session_token")
    user = None
    if session_token:
        user = validate_session(session_token, db=database)
        if not user:
            # Token expired or invalid -> clear
            st.session_state.clear()
            st.rerun()

    # If unauthenticated, render Auth View (Login / Register)
    if not user:
        render_auth_view(db=database)
        return

    # Authenticated User Session - initialize worker and search engine
    search_wrapper = get_search_wrapper()
    ingestion_runner = get_ingestion_worker()

    # Authenticated User Session
    st.session_state["user"] = user

    # Render Sidebar and determine active tab
    active_view = render_sidebar(user, db=database)

    # Background status fragment
    poll_pipeline_status()

    # Main view routing
    if active_view == "search":
        render_search_view(search_wrapper, user, db=database)
    elif active_view == "upload":
        render_upload_view(ingestion_runner, user, db=database)
    elif active_view == "library":
        render_library_view(ingestion_runner, user, db=database)
    else:
        render_search_view(search_wrapper, user, db=database)


if __name__ == "__main__":
    main()
