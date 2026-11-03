"""Run in .venv-airflow: ``.venv-airflow/bin/python -m pytest airflow/tests``. Airflow reads its home when imported,
so a temp home is set first (nothing touches ~/airflow or airflow_home/)."""

import os
import tempfile

os.environ["AIRFLOW_HOME"] = tempfile.mkdtemp(prefix="airflow-home-")
os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
