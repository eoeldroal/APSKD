import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from verl.tools.schemas import OpenAIFunctionToolSchema


def _tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "computer_13",
                "description": "Execute ordered GUI actions in OSWorld computer_13.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "description": "Ordered list of computer_13 actions.",
                        }
                    },
                    "required": ["actions"],
                },
            },
        }
    )


def _png_base64() -> str:
    image = Image.new("RGB", (3, 2), (0, 255, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_computer_13_tool_posts_request_id_task_id_and_actions(monkeypatch):
    from verl.tools.computer_13_tool import Computer13Tool

    captured = {}

    async def fake_post_json(self, payload):
        captured["url"] = self.url
        captured["timeout"] = self.timeout
        captured["payload"] = payload
        return {
            "request_id": "req-trajectory",
            "task_id": "osworld-000421",
            "status": "ok",
            "text": "A11Y_TREE:\n- window: test",
            "image": {
                "data": _png_base64(),
                "mimeType": "image/png",
            },
        }

    monkeypatch.setattr(Computer13Tool, "_post_json", fake_post_json)

    tool = Computer13Tool(
        config={"url": "http://127.0.0.1:8000/computer_13/call", "timeout": 30},
        tool_schema=_tool_schema(),
    )
    instance_id, _ = await tool.create(create_kwargs={"task_id": "osworld-000421"})

    actions = [
        {"action_type": "MOVE_TO", "x": 420, "y": 315},
        {"action_type": "CLICK", "button": "left", "x": 420, "y": 315, "num_clicks": 1},
    ]
    response, reward, metrics = await tool.execute(
        instance_id,
        {"actions": actions},
        agent_data=SimpleNamespace(request_id="req-trajectory"),
    )

    assert captured["url"] == "http://127.0.0.1:8000/computer_13/call"
    assert captured["timeout"] == 30
    assert captured["payload"] == {
        "request_id": "req-trajectory",
        "task_id": "osworld-000421",
        "actions": actions,
    }
    assert reward == 0.0
    assert response.text == "A11Y_TREE:\n- window: test"
    assert response.image is not None
    assert len(response.image) == 1
    assert response.image[0].mode == "RGB"
    assert response.image[0].size == (3, 2)
    assert metrics == {
        "status": "ok",
        "task_id": "osworld-000421",
        "api_request_error": None,
        "image_count": 1,
    }


@pytest.mark.asyncio
async def test_computer_13_tool_returns_text_only_error_response(monkeypatch):
    from verl.tools.computer_13_tool import Computer13Tool

    async def fake_post_json(self, payload):
        del self, payload
        return {
            "request_id": "req-trajectory",
            "task_id": "osworld-000421",
            "status": "error",
            "text": "ERROR: action 2 failed.",
        }

    monkeypatch.setattr(Computer13Tool, "_post_json", fake_post_json)

    tool = Computer13Tool(config={"url": "http://server/call"}, tool_schema=_tool_schema())
    instance_id, _ = await tool.create(create_kwargs={"task_id": "osworld-000421"})

    response, reward, metrics = await tool.execute(
        instance_id,
        {"actions": [{"action_type": "WAIT"}]},
        agent_data=SimpleNamespace(request_id="req-trajectory"),
    )

    assert reward == 0.0
    assert response.text == "ERROR: action 2 failed."
    assert response.image is None
    assert metrics["status"] == "error"
    assert metrics["api_request_error"] is None
    assert metrics["image_count"] == 0


@pytest.mark.asyncio
async def test_computer_13_tool_rejects_fields_outside_action_space(monkeypatch):
    from verl.tools.computer_13_tool import Computer13Tool

    called = False

    async def fake_post_json(self, payload):
        nonlocal called
        del self, payload
        called = True
        return {}

    monkeypatch.setattr(Computer13Tool, "_post_json", fake_post_json)

    tool = Computer13Tool(config={"url": "http://server/call"}, tool_schema=_tool_schema())
    instance_id, _ = await tool.create(create_kwargs={"task_id": "osworld-000421"})

    response, reward, metrics = await tool.execute(
        instance_id,
        {"actions": [{"action_type": "WAIT", "seconds": 1}]},
        agent_data=SimpleNamespace(request_id="req-trajectory"),
    )

    assert called is False
    assert reward == 0.0
    assert response.text.startswith("ERROR:")
    assert "unexpected fields" in response.text
    assert response.image is None
    assert metrics["api_request_error"] == response.text
