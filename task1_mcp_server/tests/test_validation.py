"""Schema validation + JSON-RPC error mapping, exercised over a real stdio subprocess."""
import json

import pytest

INVALID_PARAMS = -32602
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601

GOOD_REASON = "Item arrived damaged"


def assert_invalid_params(resp, *, field=None):
    assert "result" not in resp, f"expected JSON-RPC error, got result: {resp}"
    assert resp["error"]["code"] == INVALID_PARAMS, resp
    if field:
        fields = {e["field"] for e in resp["error"]["data"]["errors"]}
        assert field in fields, f"expected error on {field!r}, got {fields}"


# ---------------------------------------------------------------- discovery
def test_tools_list_exposes_both_tools_with_schemas(client):
    resp = client.request("tools/list")
    tools = {t["name"]: t for t in resp["result"]["tools"]}
    assert set(tools) == {"get_customer_record", "trigger_refund"}
    assert tools["get_customer_record"]["inputSchema"]["required"] == ["customer_id"]
    assert set(tools["trigger_refund"]["inputSchema"]["required"]) == {"customer_id", "amount", "reason"}
    assert tools["trigger_refund"]["inputSchema"]["additionalProperties"] is False


# ---------------------------------------------------------------- happy path
def test_get_customer_record_ok(client):
    resp = client.call("get_customer_record", {"customer_id": "CUST-00001"})
    assert resp["result"]["isError"] is False
    assert resp["result"]["structuredContent"]["name"] == "Ada Lovelace"


@pytest.mark.parametrize("amount", [5, 19.99, 0.01, 1_000_000.5])
def test_trigger_refund_ok(client, amount):
    resp = client.call("trigger_refund", {"customer_id": "CUST-00001", "amount": amount, "reason": GOOD_REASON})
    assert resp["result"]["isError"] is False
    assert resp["result"]["structuredContent"]["status"] == "queued"
    assert resp["result"]["structuredContent"]["refund_id"].startswith("REF-")


def test_reason_exactly_ten_chars_ok(client):
    resp = client.call("trigger_refund", {"customer_id": "CUST-00001", "amount": 1, "reason": "0123456789"})
    assert "result" in resp and resp["result"]["isError"] is False


def test_business_failure_is_tool_error_not_protocol_error(client):
    """Well-formed request, unknown customer -> isError result, NOT a JSON-RPC error."""
    resp = client.call("get_customer_record", {"customer_id": "CUST-99999"})
    assert "error" not in resp
    assert resp["result"]["isError"] is True


# ---------------------------------------------------------------- customer_id
BAD_CUSTOMER_IDS = [
    "",
    "CUST-1234",  # too short
    "CUST-123456",  # too long
    "cust-12345",  # case
    "CUST-1234a",  # non-digit
    "CUST12345",  # missing dash
    "XCUST-12345",
    "CUST-12345X",
    " CUST-12345",  # leading space
    "CUST-12345 ",  # trailing space
    "CUST-12345\n",  # trailing newline ($ anchoring trap)
    "CUST-١٢٣٤٥",  # Arabic-Indic digits (\d Unicode trap)
    "CUST-１２３４５",  # full-width digits
    12345,
    None,
    ["CUST-12345"],
    {"id": "CUST-12345"},
    True,
]


@pytest.mark.parametrize("bad", BAD_CUSTOMER_IDS, ids=[repr(b)[:30] for b in BAD_CUSTOMER_IDS])
def test_get_customer_record_rejects_bad_id(client, bad):
    assert_invalid_params(client.call("get_customer_record", {"customer_id": bad}), field="customer_id")


@pytest.mark.parametrize("bad", BAD_CUSTOMER_IDS[:6], ids=[repr(b) for b in BAD_CUSTOMER_IDS[:6]])
def test_trigger_refund_rejects_bad_id(client, bad):
    resp = client.call("trigger_refund", {"customer_id": bad, "amount": 5, "reason": GOOD_REASON})
    assert_invalid_params(resp, field="customer_id")


# ---------------------------------------------------------------- amount
@pytest.mark.parametrize(
    "bad",
    [0, 0.0, -1, -0.01, "10", "10.5", True, False, None, [], {}, 0.001, 0.1 + 0.2, 1e-05],
    ids=lambda v: repr(v),
)
def test_refund_rejects_bad_amount(client, bad):
    resp = client.call("trigger_refund", {"customer_id": "CUST-00001", "amount": bad, "reason": GOOD_REASON})
    assert_invalid_params(resp, field="amount")


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_refund_rejects_non_finite_amount_from_raw_json(client, literal):
    # json.dumps would refuse nothing here, but Python's parser on the other end accepts these literals.
    line = (
        '{"jsonrpc":"2.0","id":77,"method":"tools/call","params":{"name":"trigger_refund","arguments":'
        f'{{"customer_id":"CUST-00001","amount":{literal},"reason":"{GOOD_REASON}"}}}}}}'
    )
    client.send_raw(line)
    assert_invalid_params(client.read_response(77), field="amount")


# ---------------------------------------------------------------- reason
@pytest.mark.parametrize(
    "bad",
    ["", "short", "123456789", " " * 20, "\n\t " * 8, "   ab   ", 12345678901, None, ["a long enough reason"], "x" * 501],
    ids=lambda v: repr(v)[:30],
)
def test_refund_rejects_bad_reason(client, bad):
    resp = client.call("trigger_refund", {"customer_id": "CUST-00001", "amount": 5, "reason": bad})
    assert_invalid_params(resp, field="reason")


# ---------------------------------------------------------------- structure
def test_missing_arguments_reports_every_missing_field(client, missing):
    resp = client.call("trigger_refund", missing)
    assert_invalid_params(resp)
    assert {e["field"] for e in resp["error"]["data"]["errors"]} == {"customer_id", "amount", "reason"}


def test_all_errors_reported_at_once(client):
    resp = client.call("trigger_refund", {"customer_id": "bad", "amount": -1, "reason": "no"})
    assert {e["field"] for e in resp["error"]["data"]["errors"]} == {"customer_id", "amount", "reason"}


def test_extra_fields_rejected(client):
    resp = client.call("get_customer_record", {"customer_id": "CUST-00001", "admin": True})
    assert_invalid_params(resp, field="admin")


@pytest.mark.parametrize("args", ["CUST-00001", [], 5, None], ids=repr)
def test_non_object_arguments_rejected(client, args):
    resp = client.call("get_customer_record", args)
    assert "result" not in resp and resp["error"]["code"] == INVALID_PARAMS


def test_unknown_tool_is_invalid_params(client):
    resp = client.call("delete_everything", {})
    assert resp["error"]["code"] == INVALID_PARAMS
    assert "delete_everything" in resp["error"]["message"]


def test_unknown_method(client):
    assert client.request("nope/nope")["error"]["code"] == METHOD_NOT_FOUND


def test_validation_errors_do_not_echo_input(client):
    secret = "SECRET-VALUE-4111111111111111"
    resp = client.call("get_customer_record", {"customer_id": secret})
    assert secret not in json.dumps(resp)


# ---------------------------------------------------------------- malformed wire input
@pytest.mark.parametrize(
    "line,code",
    [
        ("this is not json", PARSE_ERROR),
        ('{"jsonrpc":"2.0","id":5,"method":"tools/li', PARSE_ERROR),
        ('{"foo": 1}', INVALID_REQUEST),
        ('[{"jsonrpc":"2.0","id":1,"method":"ping"}]', INVALID_REQUEST),
        ('{"jsonrpc":"1.0","id":1,"method":"ping"}', INVALID_REQUEST),
    ],
    ids=["garbage", "truncated", "not-jsonrpc", "batch", "wrong-version"],
)
def test_malformed_wire_input_gets_standard_error_with_null_id(client, line, code):
    client.send_raw(line)
    resp = json.loads(client.recv_line())
    assert resp["id"] is None and resp["error"]["code"] == code


def test_server_survives_malformed_input(client):
    client.send_raw("}{ garbage")
    assert json.loads(client.recv_line())["error"]["code"] == PARSE_ERROR
    resp = client.call("get_customer_record", {"customer_id": "CUST-00001"})
    assert resp["result"]["isError"] is False
