"""Mock OpenAI-compatible streaming provider. Body extras: x_text, x_chunk (chars/delta), x_delay (s/delta)."""
import asyncio, json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
DEFAULT = "Sure! Reach Jane at jane.doe@example.com. Her SSN is 123-45-6789 and card 4111 1111 1111 1111. Anything else?"


@app.post("/v1/chat/completions")
async def completions(request: Request):
    b = await request.json()
    text, n, delay = b.get("x_text", DEFAULT), int(b.get("x_chunk", 3)), float(b.get("x_delay", 0))

    async def gen():
        base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "mock"}
        yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n"
        for i in range(0, len(text), n):
            if delay: await asyncio.sleep(delay)
            yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {"content": text[i:i+n]}, "finish_reason": None}]}) + "\n\n"
        yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
