from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any, Awaitable, Callable

import anyio
import mcp_types as types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.shared.message import SessionMessage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

logger = logging.getLogger("task1.mcp_server")

CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"
REASON_MIN_LENGTH = 10
REASON_MAX_LENGTH = 500
MAX_DECIMAL_PLACES = 2

CustomerId = Annotated[
    str,
    Field(
        strict=True,
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier, format CUST-XXXXX (five digits).",
        examples=["CUST-00001"],
    ),
]


class _StrictArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class GetCustomerRecordArgs(_StrictArgs):
    customer_id: CustomerId


class TriggerRefundArgs(_StrictArgs):
    customer_id: CustomerId
    amount: Annotated[
        float,
        Field(
            strict=True,  
            gt=0,
            allow_inf_nan=False,
            description="Refund amount, positive, at most 2 decimal places.",
        ),
    ]
    reason: Annotated[
        str,
        StringConstraints(
            strict=True,
            strip_whitespace=True,  
            min_length=REASON_MIN_LENGTH,
            max_length=REASON_MAX_LENGTH,
        ),
    ]

    @field_validator("amount")
    @classmethod
    def _no_sub_cent_precision(cls, v: float) -> float:
        exponent = Decimal(repr(v)).as_tuple().exponent
        if isinstance(exponent, int) and exponent < -MAX_DECIMAL_PLACES:
            raise ValueError(f"amount must have at most {MAX_DECIMAL_PLACES} decimal places")
        return v


_CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-00001": {"name": "Ada Lovelace", "email": "ada@example.com", "tier": "gold", "balance_owed": 0.0},
    "CUST-00002": {"name": "Grace Hopper", "email": "grace@example.com", "tier": "silver", "balance_owed": 12.5},
    "CUST-12345": {"name": "Test Customer", "email": "test@example.com", "tier": "bronze", "balance_owed": 0.0},
}
_REFUNDS: list[dict[str, Any]] = []


def _ok(payload: dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload,
    )


def _tool_error(message: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)],
        is_error=True,
    )


async def get_customer_record(args: GetCustomerRecordArgs) -> types.CallToolResult:
    record = _CUSTOMERS.get(args.customer_id)
    if record is None:
        return _tool_error(f"Customer {args.customer_id} not found")
    return _ok({"customer_id": args.customer_id, **record})


async def trigger_refund(args: TriggerRefundArgs) -> types.CallToolResult:
    if args.customer_id not in _CUSTOMERS:
        return _tool_error(f"Customer {args.customer_id} not found")
    refund = {
        "refund_id": f"REF-{uuid.uuid4().hex[:12].upper()}",
        "customer_id": args.customer_id,
        "amount": args.amount,
        "reason": args.reason,
        "status": "queued",
    }
    _REFUNDS.append(refund)
    # Log ids and amounts only; free-text reason may contain personal data.
    logger.info("refund queued id=%s customer=%s amount=%s", refund["refund_id"], args.customer_id, args.amount)
    return _ok(refund)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    model: type[BaseModel]
    handler: Callable[[Any], Awaitable[types.CallToolResult]]

    def to_tool(self) -> types.Tool:
        return types.Tool(
            name=self.name,
            description=self.description,
            input_schema=self.model.model_json_schema(),
        )


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "get_customer_record",
            "Look up a customer record by customer_id (format CUST-XXXXX).",
            GetCustomerRecordArgs,
            get_customer_record,
        ),
        ToolSpec(
            "trigger_refund",
            "Queue a refund for a customer. amount must be > 0; reason must be at least "
            f"{REASON_MIN_LENGTH} characters.",
            TriggerRefundArgs,
            trigger_refund,
        ),
    )
}


def _sanitized_errors(exc: ValidationError) -> list[dict[str, Any]]:
    """Field-level problems without echoing the offending input back."""
    return [
        {"field": ".".join(str(p) for p in e["loc"]) or "(root)", "type": e["type"], "message": e["msg"]}
        for e in exc.errors(include_url=False, include_context=False, include_input=False)
    ]


async def handle_list_tools(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    return types.ListToolsResult(tools=[spec.to_tool() for spec in TOOLS.values()])


async def handle_call_tool(
    ctx: ServerRequestContext, params: types.CallToolRequestParams
) -> types.CallToolResult:
    spec = TOOLS.get(params.name)
    if spec is None:
        raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown tool: {params.name!r}")

    try:
        args = spec.model.model_validate(params.arguments or {})
    except ValidationError as exc:
        logger.warning("rejected call to %s: %d validation error(s)", spec.name, exc.error_count())
        raise MCPError(
            code=types.INVALID_PARAMS,
            message=f"Invalid arguments for tool {spec.name!r}",
            data={"errors": _sanitized_errors(exc)},
        ) from None

    try:
        return await spec.handler(args)
    except MCPError:
        raise
    except Exception:
        logger.exception("unhandled error in tool %s", spec.name)
        raise MCPError(code=types.INTERNAL_ERROR, message="Internal server error") from None


def build_server() -> Server:
    return Server(
        "customer-support-mcp",
        version="1.0.0",
        on_list_tools=handle_list_tools,
        on_call_tool=handle_call_tool,
    )


def _protocol_error_for(exc: Exception) -> SessionMessage:
    is_bad_json = (
        isinstance(exc, ValidationError) and bool(exc.errors()) and exc.errors()[0]["type"] == "json_invalid"
    )
    code, message = (
        (types.PARSE_ERROR, "Parse error") if is_bad_json else (types.INVALID_REQUEST, "Invalid Request")
    )
    return SessionMessage(
        types.JSONRPCError(jsonrpc="2.0", id=None, error=types.ErrorData(code=code, message=message))
    )


async def serve() -> None:
    server = build_server()
    async with stdio_server() as (transport_read, transport_write):
        to_server, server_read = anyio.create_memory_object_stream[SessionMessage | Exception](0)

        async def pump() -> None:
            async with to_server:
                async for item in transport_read:
                    try:
                        if isinstance(item, Exception):
                            logger.warning("unparseable input line: %s", type(item).__name__)
                            await transport_write.send(_protocol_error_for(item))
                        else:
                            await to_server.send(item)
                    except (anyio.ClosedResourceError, anyio.BrokenResourceError):
                        return

        async with anyio.create_task_group() as tg:
            tg.start_soon(pump)
            await server.run(server_read, transport_write, server.create_initialization_options())
            tg.cancel_scope.cancel()


def configure_logging() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=os.environ.get("MCP_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    configure_logging()
    logger.info("starting customer-support-mcp on stdio (pid=%d)", os.getpid())
    anyio.run(serve)


if __name__ == "__main__":
    main()
