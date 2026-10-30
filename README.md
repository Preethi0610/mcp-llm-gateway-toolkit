# AI Solutions Engineer Assessment

Four small python projects, one folder each, each with its own tests. No API keys needed, I used mock servers instead of a real LLM.

## How to run

Python 3.11+ (I used 3.13)

```
python3 -m venv .venv
source .venv/bin/activate      (windows: .venv\Scripts\activate)
pip install -r requirements.txt
./run_all_tests.sh
```

Tests run per folder because task 2 and task 3 both have a gateway.py.

## Task 1 - MCP server (task1_mcp_server)

Two tools, get_customer_record and trigger_refund, using the official mcp sdk over stdio. Inputs are checked with Pydantic (strict, no extra fields). Bad input returns a -32602 error and doesn't echo the value back.

Nothing prints to stdout, logs go to stderr. I added a test that prints inside a tool on purpose to make sure the output stays clean JSON-RPC.

Things I assumed: customer id is CUST- plus exactly 5 digits, and refund amount can have max 2 decimals (my own addition).

## Task 2 - MCP gateway (task2_mcp_gateway)

Proxy that reads the Bearer token to get the role. tools/list just gets forwarded. If tools/call is for a tool starting with admin_ and the role isn't admin, it returns -32001 right away and never calls the downstream server. Tests check that.

Also blocks batch requests, upper case ADMIN_, and alg=none JWTs. It doesn't support streaming (SSE) responses from downstream.

## Task 3 - PII redaction (task3_pii_stream)

Streams the response and replaces emails, SSNs and card numbers with [REDACTED]. The tricky part is when an email is split between chunks, so it holds back only the last unfinished word and sends the rest right away. Memory stays small however long the response is.

Card numbers must pass the Luhn check so normal numbers don't get redacted. Only handles regular text content, and regex can't catch names or addresses.

## Task 4 - Rate limiter and fallback (task4_router)

50,000 tokens per minute per key, saved in a sqlite file. The check and insert happen in one transaction so parallel requests can't go over the limit. Only a hash of the key is stored.

If the primary model gives a 429 or takes longer than 3 seconds it switches to the backup (I also do this for 5xx errors). All errors return the same format with a request id, and internal details only go to the logs.
