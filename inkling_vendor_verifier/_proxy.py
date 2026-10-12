"""HTTP CONNECT proxy for the task container's isolated Docker network."""

from __future__ import annotations

import http.client
import ipaddress
import json
import select
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def permitted_host(host: str, allowed_hosts: list[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(pattern == "*" or host == pattern.lower().rstrip(".") for pattern in allowed_hosts)


def connect(host: str, port: int, allowed_hosts: list[str]) -> socket.socket:
    if port not in (80, 443) or not permitted_host(host, allowed_hosts):
        raise PermissionError("Destination is not allowed")
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise PermissionError("Only public destination addresses are allowed")
    last_error: OSError | None = None
    for family, socktype, protocol, _, address in addresses:
        connection = socket.socket(family, socktype, protocol)
        connection.settimeout(30)
        try:
            connection.connect(address)
        except OSError as error:
            connection.close()
            last_error = error
        else:
            return connection
    assert last_error is not None
    raise last_error


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    policy_file: Path

    def _connection(self, host: str, port: int) -> socket.socket:
        return connect(host, port, json.loads(self.policy_file.read_text()))

    def do_CONNECT(self) -> None:
        target = urlsplit("//" + self.path)
        try:
            if target.hostname is None or target.username is not None:
                raise PermissionError("Invalid CONNECT destination")
            upstream = self._connection(target.hostname, target.port or 443)
        except (PermissionError, ValueError):
            self.send_error(403)
            return
        except OSError:
            self.send_error(502)
            return
        with upstream:
            self.close_connection = True
            self.send_response(200, "Connection Established")
            self.end_headers()
            self.connection.settimeout(60)
            while True:
                readable, _, _ = select.select((self.connection, upstream), (), (), 60)
                if not readable:
                    break
                for source in readable:
                    try:
                        data = source.recv(65536)
                        if not data:
                            return
                        destination = upstream if source is self.connection else self.connection
                        destination.sendall(data)
                    except OSError:
                        return

    def _http(self) -> None:
        target = urlsplit(self.path)
        try:
            if target.scheme != "http" or target.hostname is None or target.username is not None:
                raise PermissionError("Expected an absolute HTTP URL")
            if self.headers.get("Transfer-Encoding") is not None:
                raise PermissionError("Chunked request bodies are unsupported")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 16 * 1024 * 1024:
                raise PermissionError("Request body is too large")
            upstream = self._connection(target.hostname, target.port or 80)
        except (PermissionError, ValueError):
            self.send_error(403)
            return
        except OSError:
            self.send_error(502)
            return
        client = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=60)
        client.sock = upstream
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "connection", "proxy-connection", "proxy-authorization"}
        }
        headers["Host"] = target.netloc
        headers["Connection"] = "close"
        try:
            client.request(
                self.command,
                (target.path or "/") + ("?" + target.query if target.query else ""),
                body=self.rfile.read(length) if length else None,
                headers=headers,
            )
            response = client.getresponse()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in {"transfer-encoding", "connection", "content-length"}:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while data := response.read(65536):
                self.wfile.write(data)
        except OSError:
            self.close_connection = True
        finally:
            client.close()
            self.close_connection = True

    do_GET = _http
    do_HEAD = _http
    do_POST = _http
    do_PUT = _http
    do_DELETE = _http


if __name__ == "__main__":
    Proxy.policy_file = Path(sys.argv[1])
    ThreadingHTTPServer(("0.0.0.0", 3128), Proxy).serve_forever()
