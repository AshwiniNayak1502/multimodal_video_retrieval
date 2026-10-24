"""Centralized offline environment configuration for Hugging Face and PyTorch models."""

import os


def configure_offline_env() -> None:
    """Ensure Hugging Face and Transformers strictly operate offline without network calls."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


# Configure immediately upon import
configure_offline_env()
