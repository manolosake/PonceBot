#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import secrets
from pathlib import Path


def pbkdf2_sha256(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create/update codexbot users.json with salted password hashes.")
    ap.add_argument("--users-file", default=str(Path(__file__).resolve().parents[1] / "users.json"))
    ap.add_argument("--username", required=True)
    ap.add_argument("--profile", required=True, help="Profile name (must exist in profiles.json)")
    ap.add_argument("--iterations", type=int, default=200_000)
    args = ap.parse_args()

    users_file = Path(args.users_file).expanduser().resolve()
    username = args.username.strip()
    profile = args.profile.strip()
    iters = int(args.iterations)
    if not username:
        raise SystemExit("username is empty")
    if not profile:
        raise SystemExit("profile is empty")
    if iters < 50_000:
        raise SystemExit("iterations too low")

    pw1 = getpass.getpass("Password: ")
    pw2 = getpass.getpass("Password (again): ")
    if pw1 != pw2:
        raise SystemExit("Passwords do not match")
    if not pw1:
        raise SystemExit("Empty password not allowed")

    salt = secrets.token_bytes(16)
    dk = pbkdf2_sha256(pw1, salt, iters)
    rec = {
        "profile": profile,
        "salt_b64": base64.b64encode(salt).decode("ascii"),
        "hash_b64": base64.b64encode(dk).decode("ascii"),
        "iterations": iters,
    }

    data = load_json(users_file)
    if not isinstance(data, dict):
        data = {}
    users = data.get("users")
    if not isinstance(users, dict):
        users = {}
    users[username] = rec
    data["version"] = 1
    data["users"] = users
    atomic_write_json(users_file, data)

    try:
        users_file.chmod(0o600)
    except Exception:
        pass

    print(f"OK. Wrote {users_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

