# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import binascii
from io import BytesIO
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import aiohttp
from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class Computer13Tool(BaseTool):
    """Native HTTP adapter for the OSWorld computer_13 action server."""

    _SUPPORTED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg"}
    _ACTION_FIELDS: dict[str, set[str]] = {
        "MOVE_TO": {"action_type", "x", "y"},
        "CLICK": {"action_type", "button", "x", "y", "num_clicks"},
        "MOUSE_DOWN": {"action_type", "button"},
        "MOUSE_UP": {"action_type", "button"},
        "RIGHT_CLICK": {"action_type", "x", "y"},
        "DOUBLE_CLICK": {"action_type", "x", "y"},
        "DRAG_TO": {"action_type", "x", "y"},
        "SCROLL": {"action_type", "dx", "dy"},
        "TYPING": {"action_type", "text"},
        "PRESS": {"action_type", "key"},
        "KEY_DOWN": {"action_type", "key"},
        "KEY_UP": {"action_type", "key"},
        "HOTKEY": {"action_type", "keys"},
        "WAIT": {"action_type"},
        "FAIL": {"action_type"},
        "DONE": {"action_type"},
    }

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.url = config.get("url", "")
        if not self.url:
            raise ValueError("Computer13Tool requires config.url")
        self.timeout = float(config.get("timeout", 30))
        self.max_image_bytes = int(config.get("max_image_bytes", 10 * 1024 * 1024))
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        task_id = create_kwargs.get("task_id")
        if not task_id:
            raise ValueError("Computer13Tool requires create_kwargs.task_id")
        self._instance_dict[instance_id] = {"task_id": str(task_id)}
        return instance_id, ToolResponse()

    async def release(self, instance_id: str, **kwargs) -> None:
        del kwargs
        self._instance_dict.pop(instance_id, None)

    async def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.url, json=payload) as response:
                response.raise_for_status()
                return await response.json()

    def _validate_actions(self, actions: Any) -> list[dict[str, Any]]:
        if not isinstance(actions, list) or not actions:
            raise ValueError("actions must be a non-empty list")

        validated: list[dict[str, Any]] = []
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                raise ValueError(f"action {index} must be an object")
            action_type = action.get("action_type")
            if action_type not in self._ACTION_FIELDS:
                raise ValueError(f"action {index} has unsupported action_type: {action_type!r}")

            allowed_fields = self._ACTION_FIELDS[action_type]
            unexpected_fields = sorted(set(action) - allowed_fields)
            if unexpected_fields:
                raise ValueError(
                    f"action {index} ({action_type}) has unexpected fields: {', '.join(unexpected_fields)}"
                )
            validated.append(dict(action))

        return validated

    def _decode_image(self, image_payload: Any) -> Image.Image:
        if not isinstance(image_payload, dict):
            raise ValueError("image must be an object")
        mime_type = image_payload.get("mimeType")
        if mime_type not in self._SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError(f"Unsupported image mimeType: {mime_type!r}")
        data = image_payload.get("data")
        if not data:
            raise ValueError("image.data is missing")

        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image.data is not valid base64") from exc
        if len(raw) > self.max_image_bytes:
            raise ValueError(f"image exceeds max_image_bytes: {len(raw)} > {self.max_image_bytes}")

        image = Image.open(BytesIO(raw))
        image.load()
        return image.convert("RGB")

    @staticmethod
    def _tool_response_kwargs(text: str | None, image: Image.Image | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"text": text}
        if image is not None:
            kwargs["image"] = [image]
        return kwargs

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        request_id = getattr(agent_data, "request_id", None)
        if not request_id:
            error_msg = "ERROR: Computer13Tool requires agent_data.request_id"
            return ToolResponse(text=error_msg), 0.0, {"api_request_error": error_msg, "image_count": 0}
        if instance_id not in self._instance_dict:
            error_msg = f"ERROR: unknown Computer13Tool instance_id: {instance_id}"
            return ToolResponse(text=error_msg), 0.0, {"api_request_error": error_msg, "image_count": 0}

        try:
            actions = self._validate_actions(parameters.get("actions"))
            task_id = self._instance_dict[instance_id]["task_id"]
            payload = {"request_id": str(request_id), "task_id": task_id, "actions": actions}
            result = await self._post_json(payload)

            text = result.get("text")
            if text is None:
                text = ""
            image = self._decode_image(result["image"]) if "image" in result else None
            status = result.get("status", "unknown")
            response_task_id = result.get("task_id", task_id)

            return (
                ToolResponse(**self._tool_response_kwargs(str(text), image)),
                0.0,
                {
                    "status": status,
                    "task_id": response_task_id,
                    "api_request_error": None,
                    "image_count": 1 if image is not None else 0,
                },
            )
        except Exception as exc:
            error_msg = f"ERROR: Computer13Tool execution failed: {exc}"
            logger.error("[Computer13Tool] Execution failed: %s", exc)
            return ToolResponse(text=error_msg), 0.0, {"api_request_error": error_msg, "image_count": 0}
