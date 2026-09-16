"""Authenticated, certificate-pinned cluster transport for one learner.

The server is bound explicitly to the private Spark link, never all interfaces.
Numeric NPZ only (no pickle), bounded uploads, immutable checkpoint versions,
fixed-interface/run validation, duplicate IDs and stale rollout rejection.
"""
import hashlib
import hmac
import io
import json
import re
import secrets
import ssl
import threading
import urllib.request
import zipfile
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
from .layout_recovery_world import CHANNELS, INTERFACE, PHYSICS

LIMIT = 48*1024*1024


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def pack_rollout(observations, labels, state, meta):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, observations=observations, labels=labels, state=state,
                        metadata=np.asarray(json.dumps(meta)))
    return buffer.getvalue()


def unpack_rollout(blob):
    if len(blob) > LIMIT:
        raise ValueError('Oversize rollout')
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        if sum(info.file_size for info in archive.infolist()) > LIMIT:
            raise ValueError('Expanded rollout too large')
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        x, y, state = (data[name].copy() for name in ('observations', 'labels', 'state'))
        meta = json.loads(str(data['metadata']))
    if x.ndim != 3 or x.shape[2] != CHANNELS or not 1 <= x.shape[1] <= 16 or not 32 <= x.shape[0] <= 256:
        raise ValueError('Wrong observation shape')
    if y.shape != (*x.shape[:2], 3) or state.shape != (166700, x.shape[1]):
        raise ValueError('Wrong label/state shape')
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (x, y, state)):
        raise ValueError('Rollout must be finite float32')
    if meta.get('interface') != INTERFACE or meta.get('physics') != PHYSICS or meta.get('control') != 'learner-only':
        raise ValueError('Unapproved rollout interface/control')
    if not re.fullmatch(r'[a-f0-9]{32}', meta.get('id', '')):
        raise ValueError('Invalid rollout id')
    if not isinstance(meta.get('version'), int) or meta['version'] < 0:
        raise ValueError('Invalid version')
    return x, y, state, meta


class Exchange:
    def __init__(self, out, token, fixed_hash, run_id=None, max_queue=24, layout_hash=None):
        self.out, self.token, self.fixed_hash = Path(out), token, fixed_hash
        self.run_id = run_id or secrets.token_hex(16)
        self.lock = threading.Lock()
        self.queue = deque()
        self.seen = set()
        self.versions = {}
        self.version = 0
        self.update = 0
        self.finished = False
        self.max_queue = max_queue
        self.received = self.consumed = 0
        self.last_worker = None
        self.layout_hash = layout_hash

    def publish(self, version, filename, parameter_hash):
        with self.lock:
            self.versions[version] = {'file': filename, 'parameterHash': parameter_hash,
                                      'sha256': hashlib.sha256((self.out/filename).read_bytes()).hexdigest()}
            self.version = version

    def manifest(self):
        with self.lock:
            return {'runId': self.run_id, 'version': self.version, 'update': self.update,
                    'interface': INTERFACE, 'physics': PHYSICS, 'fixedHash': self.fixed_hash,
                    'checkpoint': self.versions[self.version], 'finished': self.finished,
                    'queued': len(self.queue), 'received': self.received, 'consumed': self.consumed,
                    'lastWorker': self.last_worker, 'layoutHash': self.layout_hash}

    def accept(self, blob):
        value = unpack_rollout(blob)
        meta = value[3]
        with self.lock:
            if meta.get('runId') != self.run_id or meta.get('fixedHash') != self.fixed_hash:
                raise ValueError('Different experiment/interface')
            if self.layout_hash is not None and (meta.get('layoutHash') != self.layout_hash
                    or not meta.get('worlds')
                    or any(w.get('layoutHash') != self.layout_hash for w in meta['worlds'])):
                raise ValueError('Different fixed-layout training task')
            if meta['version'] not in self.versions or meta.get('parameterHash') != self.versions[meta['version']]['parameterHash']:
                raise ValueError('Unrecognized checkpoint')
            if self.update-meta['version'] > 80:
                return 'stale'
            if meta['id'] in self.seen:
                return 'duplicate'
            if self.finished:
                return 'finished'
            if len(self.queue) >= self.max_queue:
                return 'full'
            # Save raw evidence before acknowledgement. The existing directory
            # is an experiment artifact, never a general upload destination.
            target = self.out/'experience'/f"{meta['id']}.npz"
            target.write_bytes(blob)
            self.seen.add(meta['id'])
            self.queue.append(target)
            self.received += 1
            self.last_worker = {'id': meta.get('worker'), 'version': meta['version'], 'rollout': meta['id']}
            return 'accepted'

    def pop(self):
        with self.lock:
            path = self.queue.popleft() if self.queue else None
        if path is None:
            return None
        value = unpack_rollout(path.read_bytes())
        if self.update-value[3]['version'] > 80:
            return None
        self.consumed += 1
        return value


from .deployment import COORDINATOR_HOST, COORDINATOR_URL


def start_server(exchange, host, port, cert, key):
    if host != COORDINATOR_HOST:
        raise ValueError('Server must bind only the private Spark cluster link')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def authenticated(self):
            return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+exchange.token)

        def reply(self, status, body, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.authenticated():
                return self.reply(401, b'{}')
            if self.path == '/manifest':
                return self.reply(200, json.dumps(exchange.manifest()).encode())
            match = re.fullmatch(r'/checkpoint/(\d+)', self.path)
            if match and int(match[1]) in exchange.versions:
                path = exchange.out/exchange.versions[int(match[1])]['file']
                return self.reply(200, path.read_bytes(), 'application/octet-stream')
            return self.reply(404, b'{}')

        def do_POST(self):
            if not self.authenticated():
                return self.reply(401, b'{}')
            if self.path != '/rollout':
                return self.reply(404, b'{}')
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= LIMIT:
                    return self.reply(413, b'{}')
                self.connection.settimeout(90)
                blob = self.rfile.read(length)
                if len(blob) != length:
                    raise ValueError('Incomplete upload')
                result = exchange.accept(blob)
                return self.reply(200, json.dumps({'result': result}).encode())
            except (ValueError, KeyError, TypeError, zipfile.BadZipFile):
                return self.reply(400, b'{}')

    server = ThreadingHTTPServer((host, port), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert), keyfile=str(key))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class Client:
    def __init__(self, url, token_file, cert):
        if url != COORDINATOR_URL:
            raise ValueError('Only the private coordinator endpoint is allowed')
        self.url = url
        self.token = Path(token_file).read_text().strip()
        self.context = ssl.create_default_context(cafile=str(cert))
        # The private certificate is pinned by the CA file. Its hostname is an
        # experiment label rather than a DNS name; certificate validation stays on.
        self.context.check_hostname = False

    def request(self, path, data=None):
        request = urllib.request.Request(self.url+path, data=data,
                                         headers={'Authorization': 'Bearer '+self.token})
        with urllib.request.urlopen(request, context=self.context, timeout=120) as response:
            return response.read()

    def manifest(self):
        return json.loads(self.request('/manifest'))

    def upload(self, data):
        return json.loads(self.request('/rollout', data))['result']
