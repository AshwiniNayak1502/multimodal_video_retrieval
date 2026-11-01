# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build specification for VideoContextEngine."""

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, copy_metadata, collect_submodules

block_cipher = None
try:
    PROJECT_ROOT = Path(SPECPATH).resolve()
except NameError:
    PROJECT_ROOT = Path.cwd().resolve()

# 1. Collect Project Data Folders (Only production code, excluding test caches and virtualenvs)
datas = [
    (str(PROJECT_ROOT / "app"), "app"),
    (str(PROJECT_ROOT / "module1" / "src"), "module1/src"),
    (str(PROJECT_ROOT / "module1" / "main.py"), "module1"),
    (str(PROJECT_ROOT / "module2" / "embedding_store"), "module2/embedding_store"),
    (str(PROJECT_ROOT / "module2" / "main.py"), "module2"),
    (str(PROJECT_ROOT / "module3" / "retrieval_engine"), "module3/retrieval_engine"),
    (str(PROJECT_ROOT / "module3" / "main.py"), "module3"),
    (str(PROJECT_ROOT / "module3" / "config.json"), "module3"),
    (str(PROJECT_ROOT / "assets"), "assets"),
]

# Add bundled bin/ if present
bin_dir = PROJECT_ROOT / "bin"
if bin_dir.is_dir():
    datas.append((str(bin_dir), "bin"))

# 2. Collect Package Data Files and Metadata
packages_to_collect_data = [
    "streamlit",
    "torch",
    "transformers",
    "sentence_transformers",
    "whisperx",
    "faster_whisper",
    "ctranslate2",
]

for pkg in packages_to_collect_data:
    try:
        datas += collect_data_files(pkg)
    except Exception as e:
        print(f"Warning: collect_data_files for {pkg}: {e}")

metadata_packages = [
    "streamlit",
    "torch",
    "tqdm",
    "regex",
    "requests",
    "packaging",
    "filelock",
    "numpy",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    "transformers",
    "sentence_transformers",
    "whisperx",
    "faster_whisper",
    "ctranslate2",
    "scipy",
    "scikit-learn",
]

for pkg in metadata_packages:
    try:
        datas += copy_metadata(pkg)
    except Exception as e:
        print(f"Warning: copy_metadata for {pkg}: {e}")

# 3. Explicit and Collected Hidden Imports
hiddenimports = [
    # Streamlit core
    "streamlit",
    "streamlit.web.cli",
    "streamlit.runtime",
    "streamlit.runtime.scriptrunner.magic_funcs",
    "streamlit.components.v1",
    
    # ML & Torch ecosystem
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "sentence_transformers",
    "huggingface_hub",
    "safetensors",
    "faster_whisper",
    "ctranslate2",
    "whisperx",
    
    # Vector store & vision
    "faiss",
    "cv2",
    "av",
    "PIL",
    "PIL.Image",
    "scipy",
    "sklearn",
    "sklearn.utils",
    
    # Security, auth & networking
    "argon2",
    "argon2.low_level",
    "bcrypt",
    "dotenv",
    "google.oauth2.id_token",
    "google.auth.transport.requests",
    "keyring",
    "keyring.backends.Windows",
    
    # Desktop UI
    "webview",
    "webview.platforms.winforms",
    "clr_loader",
    "pythonnet",
]

# Collect submodules for internal app and key libraries
for mod in ["app", "module1", "module2", "module3", "streamlit"]:
    try:
        hiddenimports += collect_submodules(mod)
    except Exception as e:
        print(f"Warning: collect_submodules for {mod}: {e}")

hiddenimports = sorted(list(set(hiddenimports)))

# 4. Binary and Dependency Exclusions
excludes = [
    "tkinter",
    "matplotlib",
    "IPython",
    "notebook",
    "jupyter",
    "pytest",
    "_pytest",
]

a = Analysis(
    ["desktop_launcher.py"],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VideoContextEngine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # Windowed GUI application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "assets" / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VideoContextEngine",
)
