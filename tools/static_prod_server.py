#!/usr/bin/env python3
"""Serve PonceBot static production previews from /home/aponce/production-sites."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/aponce/production-sites/current"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    handler = partial(NoCacheHandler, directory=str(args.root))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving {args.root} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
