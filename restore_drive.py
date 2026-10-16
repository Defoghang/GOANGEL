import base64
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = os.environ.get("GITHUB_REPOSITORY", "Defoghang/GOANGEL")
TOKEN = os.environ.get("GH_TOKEN", "") or os.environ.get("GITHUB_TOKEN", "")
RCLONE_CONF_B64 = os.environ.get("RCLONE_CONFIG", "")
RCLONE_CONF = "/tmp/rclone.conf"
BRANCH = "cache-auto"
DRIVE_PATH = "gdrive:GOANGEL/cache"

if not TOKEN:
    print("❌ Token manquant")
    sys.exit(1)

# 1. Config rclone
if RCLONE_CONF_B64:
    Path(RCLONE_CONF).write_bytes(base64.b64decode(RCLONE_CONF_B64))
    os.chmod(RCLONE_CONF, 0o600)
    print(f"✅ rclone.conf créé ({os.path.getsize(RCLONE_CONF)} bytes)")

# 2. Liste des fichiers via API tree
print(f"── Récupération tree {BRANCH} ──")
req = urllib.request.Request(
    f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1",
    headers={"Authorization": f"Bearer {TOKEN}"},
)
with urllib.request.urlopen(req, timeout=60) as resp:
    tree = json.loads(resp.read())

all_paths = [item["path"] for item in tree["tree"] if item.get("type") == "blob"]
pkl_paths = [p for p in all_paths if p.endswith(".pkl")]

print(f"Fichiers .pkl trouvés : {len(pkl_paths)}")
for p in pkl_paths:
    print(f"  - {p}")

# 3. Récupérer OIDs depuis chaque pointeur LFS
print()
print("── Extraction des OIDs ──")
manifest = []
for path in pkl_paths:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/contents/{path}?ref={BRANCH}",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="ignore")

        oid = None
        size = None
        for line in content.splitlines():
            if line.startswith("oid sha256:"):
                oid = line.split(":", 1)[1].strip()
            elif line.startswith("size "):
                size = int(line.split()[1])

        if oid and size:
            manifest.append({"path": path, "oid": oid, "size": size})
            print(f"  ✅ {path} | OID={oid[:16]}... | size={size}")
        else:
            print(f"  ⚠️  {path} — pas un pointeur LFS")
    except Exception as e:
        print(f"  ❌ {path} : {e}")

print()
print(f"Manifest : {len(manifest)} fichiers")

# 4. Télécharger chaque blob et uploader vers Drive
print()
print("── Download + Upload vers Drive ──")
ok_count = 0
for item in manifest:
    path = item["path"]
    oid = item["oid"]
    size = item["size"]

    # Batch request LFS
    payload = json.dumps({
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": oid, "size": size}],
    }).encode()

    req = urllib.request.Request(
        f"https://github.com/{REPO}.git/info/lfs/objects/batch",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            batch = json.loads(resp.read())
        href = None
        for o in batch.get("objects", []):
            a = o.get("actions", {})
            if "download" in a:
                href = a["download"]["href"]
                break
        if not href:
            print(f"  ❌ {path} : pas d'URL")
            continue

        local = Path("/tmp/restore") / path
        local.parent.mkdir(parents=True, exist_ok=True)

        with urllib.request.urlopen(href, timeout=600) as resp:
            data = resp.read()

        if len(data) != size:
            print(f"  ⚠️  {path} : taille {len(data)} vs {size}")
            continue

        local.write_bytes(data)
        print(f"  ✅ {path} téléchargé ({len(data)} bytes)")

        # Upload immédiat vers Drive
        drive_dest = f"{DRIVE_PATH}/{path[len('cache/'):]}"
        parent = os.path.dirname(drive_dest)
        subprocess.run(
            ["rclone", "--config", RCLONE_CONF, "mkdir", parent, "--parents"],
            capture_output=True, check=False,
        )
        result = subprocess.run(
            ["rclone", "--config", RCLONE_CONF, "copy", str(local), parent,
             "--stats-one-line"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            print(f"     ☁️  Drive : {drive_dest}")
            ok_count += 1
        else:
            print(f"     ❌ Drive upload : {result.stderr[:100]}")
    except Exception as e:
        print(f"  ❌ {path} : {e}")

print()
print(f"── Résumé : {ok_count}/{len(manifest)} uploadés ──")

# 5. Taille Drive
result = subprocess.run(
    ["rclone", "--config", RCLONE_CONF, "size", DRIVE_PATH],
    capture_output=True, text=True, check=False,
)
print(result.stdout)
print(result.stderr[:200] if result.stderr else "")

sys.exit(0 if ok_count > 0 else 1)
