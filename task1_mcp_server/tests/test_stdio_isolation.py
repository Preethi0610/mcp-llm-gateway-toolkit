"""STDIO isolation: stdout is JSON-RPC only; diagnostics live on stderr."""
import ast
import json
import sys
import textwrap
from pathlib import Path

from conftest import SERVER, StdioClient


# ------------------------------------------------------------ static guard
def test_source_has_no_print_or_direct_stdout_use():
    """Catches the classic mistake at review time, before it ever hits a client."""
    tree = ast.parse(SERVER.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            offenders.append(f"print() at line {node.lineno}")
        if isinstance(node, ast.Attribute) and node.attr == "stdout" and isinstance(node.value, ast.Name) and node.value.id == "sys":
            offenders.append(f"sys.stdout at line {node.lineno}")
    assert not offenders, offenders


def test_logging_is_bound_to_stderr():
    src = SERVER.read_text()
    assert "stream=sys.stderr" in src


# ------------------------------------------------------------ wire purity
def _assert_pure_jsonrpc(lines: list[str]):
    assert lines, "server produced no stdout at all"
    for raw in lines:
        assert raw.endswith("\n") and raw.count("\n") == 1, f"not newline-delimited: {raw!r}"
        msg = json.loads(raw)  # raises on any stray text
        assert isinstance(msg, dict) and msg.get("jsonrpc") == "2.0", f"not a JSON-RPC message: {raw!r}"
        assert ("result" in msg) ^ ("error" in msg) or "method" in msg, f"malformed JSON-RPC shape: {raw!r}"


def test_stdout_is_pure_jsonrpc_across_success_error_and_malformed_traffic():
    c = StdioClient()
    c.initialize()
    c.request("tools/list")
    c.call("get_customer_record", {"customer_id": "CUST-00001"})
    c.call("get_customer_record", {"customer_id": "nope"})  # validation error path (logs a warning)
    c.call("trigger_refund", {"customer_id": "CUST-00001", "amount": 12.5, "reason": "Duplicate charge on invoice"})  # logs info
    c.send_raw("garbage")  # parse error path
    c.recv_line()
    stderr = c.close()
    _assert_pure_jsonrpc(c.stdout_lines)
    # ...and the diagnostics really went somewhere: stderr.
    assert "refund queued" in stderr
    assert "rejected call" in stderr
    assert "starting customer-support-mcp" in stderr


def test_nothing_is_written_to_stdout_before_first_request():
    c = StdioClient()
    assert c.recv_line(timeout=1.0) is None, "server must stay silent on stdout until the client speaks"
    c.close()


# ------------------------------------------------------------ hostile handler
POISON = textwrap.dedent(
    f"""
    import os, sys
    from dataclasses import replace
    sys.path.insert(0, {str(SERVER.parent)!r})
    import server

    real = server.TOOLS["get_customer_record"].handler

    async def noisy(args):
        print("POISON via print")
        sys.stdout.write("POISON via sys.stdout.write\\n"); sys.stdout.flush()
        os.write(1, b"POISON via os.write(1)\\n")
        return await real(args)

    server.TOOLS["get_customer_record"] = replace(server.TOOLS["get_customer_record"], handler=noisy)
    server.main()
    """
)


def test_stray_prints_inside_a_handler_cannot_corrupt_the_wire(tmp_path: Path):
    """Defence in depth: even if a future edit (or a dependency) prints to stdout, frames stay clean."""
    script = tmp_path / "poison_server.py"
    script.write_text(POISON)
    c = StdioClient(script)
    c.initialize()
    resp = c.call("get_customer_record", {"customer_id": "CUST-00001"})
    assert resp["result"]["isError"] is False
    assert c.call("get_customer_record", {"customer_id": "CUST-00002"})["result"]["isError"] is False
    stderr = c.close()
    _assert_pure_jsonrpc(c.stdout_lines)
    assert all("POISON" not in line for line in c.stdout_lines)
    # The noise was diverted to stderr rather than lost or interleaved.
    assert stderr.count("POISON via print") >= 1
    assert "POISON via os.write(1)" in stderr
