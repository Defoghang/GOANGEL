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

KNOWN_OIDS = {
    "cache/historical.pkl": ("fa8d45e86047faa4efe758008b370b01f04a83a39bdde10e594fec348dc89b86", 1583040),
    "cache/stages/features.pkl": ("133ec2348098a6aa8b414e1f3a002e6e9aa0f5b78ec14c1dd92d08d660e8b0ca", 3331936),
}

TARGETS = [
    "cache/historical.pkl",
    "cache/stages/features.pkl",
    "cache/stages/metadata.pkl",
    "cache/stages/models/ml_XGB.pkl",
    "cache/stages/models/ml_LGBM.pkl",
    "cache/stages/models/ml_CatBoost.pkl",
    "cache/stages/models/ml_RF.pkl",
    "cache/stages/models/ml_DL.pkl",
    "cache/stages/models/pb_PoissonGoalsModel.pkl",
    "cache/stages/models/pb_DixonColesGoalModel.pkl",
    "cache/stages/models/pb_BivariatePoissonGoalModel.pkl",
    "cache/stages/models/pb_NegativeBinomialGoalModel.pkl",
    "cache/stages/models/pb_WeibullCopulaGoalsModel.pkl",
    "cache/stages/models/pb_ZeroInflatedPoissonGoalsModel.pkl",
]

if not TOKEN:
    print("❌ Token manquant")
    sys.exit(1)

if RCLONE_CONF_B64:
    Path(RCLONE_CONF).write_bytes(base64.b64decode(RCLONE_CONF_B64))
    os.chmod(RCLONE_CONF, 0o600)
    print(f"✅ rclone.conf ({os.path.getsize(RCLONE_CONF)} bytes)")


def gh_get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


# ─── 1. Récupérer tous les commits de cache-auto ───
print("── Récupération historique commits ──")
commits = []
for page in range(1, 6):
    try:
        data = gh_get(f"https://api.github.com/repos/{REPO}/commits?sha={BRANCH}&per_page=100&page={page}")
    except Exception as e:
        print(f"  Erreur page {page} : {e}")
        break
    if not data:
        break
    commits.extend(data)
    if len(data) < 100:
        break

print(f"  {len(commits)} commits trouvés")


# ─── 2. Pour chaque commit, récupérer les OIDs LFS des pkl ───
print()
print("── Extraction des OIDs LFS depuis l'historique ──")

manifest = {}

# D'abord : OIDs connus
for path, (oid, size) in KNOWN_OIDS.items():
    manifest[path] = {"oid": oid, "size": size}
    print(f"  ✅ Connu : {path}")

# Ensuite : chercher dans les commits
for idx, commit in enumerate(commits):
    sha = commit["sha"]
    if idx % 10 == 0:
        print(f"  Scan commit {idx+1}/{len(commits)} ({sha[:8]})")

    if len(manifest) >= len(TARGETS):
        break

    try:
        tree = gh_get(f"https://api.github.com/repos/{REPO}/git/trees/{sha}?recursive=1")
    except Exception:
        continue

    for item in tree.get("tree", []):
        if item.get("type") != "blob":
            continue
        path = item["path"]
        if path not in TARGETS:
            continue
        if path in manifest:
            continue

        # Récupérer le contenu pour extraire l'OID LFS
        try:
            blob = gh_get(f"https://api.github.com/repos/{REPO}/git/blobs/{item['sha']}")
            content = base64.b64decode(blob.get("content", "")).decode("utf-8", errors="ignore")
            oid = None
            size = None
            for line in content.splitlines():
                if line.startswith("oid sha256:"):
                    oid = line.split(":", 1)[1].strip()
                elif line.startswith("size "):
                    size = int(line.split()[1])
            if oid and size:
                manifest[path] = {"oid": oid, "size": size}
                print(f"  ✅ {path} (commit {sha[:8]})")
        except Exception:
            continue

print()
print(f"  Manifest : {len(manifest)}/{len(TARGETS)} fichiers")


# ─── 3. Télécharger chaque blob LFS et uploader vers Drive ───
print()
print("── Download LFS + Upload Drive ──")

ok_count = 0
for path, item in manifest.items():
    oid = item["oid"]
    size = item["size"]

    print(f"── {path} ({size} bytes) ──")

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
    except Exception as e:
        print(f"  ❌ Batch API : {e}")
        continue

    href = None
    for o in batch.get("objects", []):
        a = o.get("actions", {})
        if "download" in a:
            href = a["download"]["href"]
            break

    if not href:
        print(f"  ❌ Pas d'URL LFS (blob peut-être GC)")
        continue

    local = Path("/tmp/restore") / path
    local.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(href, timeout=600) as resp:
            data = resp.read()
        if len(data) != size:
            print(f"  ⚠️  Taille {len(data)} vs {size}")
            continue
        local.write_bytes(data)
        print(f"  ✅ Téléchargé ({len(data)} bytes)")
    except Exception as e:
        print(f"  ❌ Download : {e}")
        continue

    drive_dir = f"{DRIVE_PATH}/{os.path.dirname(path[len('cache/'):])}"
    subprocess.run(
        ["rclone", "--config", RCLONE_CONF, "mkdir", drive_dir, "--parents"],
        capture_output=True, check=False,
    )
    result = subprocess.run(
        ["rclone", "--config", RCLONE_CONF, "copy", str(local), drive_dir],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        print(f"  ☁️  Drive : {drive_dir}/{os.path.basename(path)}")
        ok_count += 1
    else:
        print(f"  ❌ Drive : {result.stderr[:200]}")


print()
print(f"── Résumé : {ok_count}/{len(manifest)} uploadés ──")

result = subprocess.run(
    ["rclone", "--config", RCLONE_CONF, "size", DRIVE_PATH],
    capture_output=True, text=True, check=False,
)
print(result.stdout)

sys.exit(0 if ok_count > 0 else 1)
