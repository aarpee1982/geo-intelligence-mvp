"""
Root entrypoint. Run with: streamlit run streamlit_app.py
This adds app/ to the Python path so utils imports resolve correctly,
then executes the main app module.
"""
import sys
from pathlib import Path

# Make app/ importable
sys.path.insert(0, str(Path(__file__).parent / "app"))

# Execute the app
exec(open(Path(__file__).parent / "app" / "app.py").read())  # noqa: S102
