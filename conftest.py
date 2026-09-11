# -*- coding: utf-8 -*-
"""Pytest bootstrap for the YouBike dispatch MVP.

Ensures the project root is importable so tests can `from src import ...`
exactly like the existing `python -m tests.validate_taskN` scripts do.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
