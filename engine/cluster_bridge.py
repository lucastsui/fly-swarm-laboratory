"""Loopback-only browser API and SSH tunnel. SSH credentials remain in memory."""
import argparse
import getpass
from http.client import HTTPException
import json
import select
import socketserver
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import paramiko

from .deployment import ORIGINS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-user", required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--api-port", type=int, default=8768)
    parser.add_argument("--tunnel-port", type=int, default=18770)
    args = parser.parse_args()
    password = getpass.getpass("Spark SSH password (memory only): ")
    token = args.token_file.read_text().strip()
    client = paramiko.SSHClient()
    client.load_host_keys(str(args.known_hosts))
    connection_lock = threading.Lock()

    def transport():
        with connection_lock:
            current = client.get_transport()
            if current is None or not current.is_active():
                client.connect(args.ssh_host, username=args.ssh_user, password=password,
                               allow_agent=False, look_for_keys=False, timeout=15)
                current = client.get_transport()
                current.set_keepalive(15)
            return current

    transport()

    class Tunnel(socketserver.BaseRequestHandler):
        def handle(self):
            channel = None
            try:
                channel = transport().open_channel("direct-tcpip", ("127.0.0.1", 8770), self.request.getpeername())
                while True:
                    readable, _, _ = select.select([self.request, channel], [], [], 30)
                    for source in readable:
                        data = source.recv(262144)
                        if not data:
                            return
                        (channel if source is self.request else self.request).sendall(data)
            except (OSError, EOFError, paramiko.SSHException):
                return
            finally:
                if channel is not None:
                    channel.close()

    class Forwarder(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    forward = Forwarder(("127.0.0.1", args.tunnel_port), Tunnel)
    threading.Thread(target=forward.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{args.tunnel_port}"

    def fetch(path, body=None):
        request = urllib.request.Request(url + path, data=body, headers={
            "Authorization": "Bearer " + token, "Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=20) as response:
                return response.status, response.read(), response.headers.get_content_type()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), "application/json"

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, code, body=b"", content_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            origin = self.headers.get("Origin", "")
            if origin in ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Private-Network", "true")
            self.end_headers()
            self.wfile.write(body)

        def handle_request(self):
            origin = self.headers.get("Origin", "")
            host = self.headers.get("Host", "").split(":")[0]
            if host not in ("127.0.0.1", "localhost") or (origin and origin not in ORIGINS):
                self.reply(403, b'{"detail":"This origin is not allowed"}')
                return
            path = urlparse(self.path).path
            if path not in ("/api/engine/state", "/api/engine/control", "/api/engine/health"):
                self.reply(404, b'{"detail":"Unknown route"}')
                return
            if self.command == "OPTIONS":
                self.reply(204)
                return
            if (self.command == "POST") != (path == "/api/engine/control"):
                self.reply(405)
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 <= size <= 16384:
                    self.reply(413)
                    return
                self.reply(*fetch(self.path, self.rfile.read(size) if self.command == "POST" else None))
            except (OSError, ValueError, HTTPException):
                self.reply(503, b'{"detail":"Spark connection unavailable; remote workers may still be training"}')

        do_GET = do_POST = do_OPTIONS = handle_request

    api = ThreadingHTTPServer(("127.0.0.1", args.api_port), Proxy)
    threading.Thread(target=api.serve_forever, daemon=True).start()
    print(f"SSH bridge ready: browser API 127.0.0.1:{args.api_port}; authenticated training tunnel {args.tunnel_port}", flush=True)
    args.storage.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            code, checkpoint, _ = fetch("/cluster/checkpoint")
            if code == 200:
                temporary = args.storage / "cluster-state.pt.tmp"
                temporary.write_bytes(checkpoint)
                temporary.replace(args.storage / "cluster-state.pt")
        except (OSError, HTTPException) as exc:
            print(f"Checkpoint backup pending: {type(exc).__name__}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
