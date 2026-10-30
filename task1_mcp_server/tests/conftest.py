from __future__ import annotations

import json
import select
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SERVER = HERE.parent / "server.py"


class StdioClient:
    def __init__(self, script: Path = SERVER) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.stdout_lines: list[str] = []  
        self._next_id = 1

    def send_raw(self, line: str) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def recv_line(self, timeout: float = 5.0) -> str | None:
        assert self.proc.stdout
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            return None
        line = self.proc.stdout.readline()
        if line == "":
            return None
        self.stdout_lines.append(line)
        return line

    def request(self, method: str, params: dict | None = None) -> dict:
        rid = self._next_id
        self._next_id += 1
        msg: dict = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self.send_raw(json.dumps(msg))
        return self.read_response(rid)

    def read_response(self, rid) -> dict:
        line = self.recv_line()
        assert line is not None, "no response from server within timeout"
        resp = json.loads(line)
        assert resp.get("id") == rid, f"response id mismatch: {resp}"
        return resp

    def call(self, name: str, arguments) -> dict:
        params: dict = {"name": name}
        if arguments is not _MISSING:
            params["arguments"] = arguments
        return self.request("tools/call", params)

    def initialize(self) -> dict:
        resp = self.request(
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}},
        )
        self.send_raw(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        return resp

    def close(self) -> str:
        """Close stdin (EOF), wait for exit, return everything the server wrote to stderr."""
        assert self.proc.stdin and self.proc.stderr
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            raise
        return self.proc.stderr.read()


_MISSING = object()


@pytest.fixture
def missing():
    return _MISSING


@pytest.fixture(scope="module")
def client():
    c = StdioClient()
    c.initialize()
    yield c
    if c.proc.poll() is None:
        c.proc.kill()
