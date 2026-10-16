#!/usr/bin/env python3
"""Encode un fichier JSON de credentials en Base64 (pour GitHub Secrets)."""
import base64
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("usage: encode_creds.py <credentials.json>", file=sys.stderr)
    sys.exit(2)

p = Path(sys.argv[1])
if not p.exists():
    print(f"introuvable: {p}", file=sys.stderr)
    sys.exit(2)

raw = p.read_bytes()
b64 = base64.b64encode(raw).decode("ascii")
print(b64)
