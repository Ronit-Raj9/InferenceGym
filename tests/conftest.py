from __future__ import annotations

import os


os.environ["LLMSERVE_MODE"] = "sim"
os.environ.pop("OPENAI_BASE_URL", None)
