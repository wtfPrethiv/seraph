"""Print which Gemini free-tier quota (if any) is currently blocking embeddings / generation."""

import re
import sys

from google import genai
from google.genai import types

client = genai.Client()
kind = sys.argv[1] if len(sys.argv) > 1 else "embed"
try:
    if kind == "embed":
        client.models.embed_content(
            model="gemini-embedding-001",
            contents=["x"],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768),
        )
    else:
        client.models.generate_content(model=sys.argv[2], contents="Say ok")
    print("ok: not rate limited right now")
except Exception as e:
    s = str(e)
    print("quota ids:", sorted(set(re.findall(r"quotaId': '([^']+)'", s))))
    print("limits:", re.findall(r"quotaValue': '([^']+)'", s), "retry:", re.findall(r"retryDelay': '([^']+)'", s))
