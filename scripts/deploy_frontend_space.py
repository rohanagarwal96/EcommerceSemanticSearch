"""One-time (or repeat-as-needed) script: push the Streamlit frontend to its
Hugging Face Space using the native Streamlit SDK (no Docker required).

Usage:
    python scripts/deploy_frontend_space.py
"""
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from ecomsearch.config import HF_SPACE_FRONTEND, HF_TOKEN, REPO_ROOT

SPACE_README = """---
title: Ecommerce Search UI
emoji: \U0001F6CD
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.60.0"
app_file: streamlit_app.py
---

Streamlit frontend for the E-Commerce Semantic Search project.
"""


def main() -> None:
    if not HF_SPACE_FRONTEND:
        raise SystemExit("HF_SPACE_FRONTEND is not set. Add it to your .env.")

    with tempfile.TemporaryDirectory() as staging:
        staging_dir = Path(staging)
        shutil.copy2(
            REPO_ROOT / "src" / "ecomsearch" / "ui" / "streamlit_app.py",
            staging_dir / "streamlit_app.py",
        )
        shutil.copy2(REPO_ROOT / "requirements-ui.txt", staging_dir / "requirements.txt")
        (staging_dir / "README.md").write_text(SPACE_README, encoding="utf-8")

        api = HfApi(token=HF_TOKEN)
        print(f"Creating (or reusing) Space '{HF_SPACE_FRONTEND}'...")
        api.create_repo(
            repo_id=HF_SPACE_FRONTEND, repo_type="space", space_sdk="streamlit", exist_ok=True
        )
        print(f"Uploading frontend to '{HF_SPACE_FRONTEND}'...")
        api.upload_folder(
            repo_id=HF_SPACE_FRONTEND,
            folder_path=str(staging_dir),
            repo_type="space",
            commit_message="Deploy frontend (native Streamlit SDK)",
        )

    print(f"Done. https://huggingface.co/spaces/{HF_SPACE_FRONTEND}")


if __name__ == "__main__":
    main()
