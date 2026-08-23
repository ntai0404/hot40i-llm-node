from __future__ import annotations

from datetime import date
from typing import Any, Sequence

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)


def _text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("message content must be text or a list of text parts")
    chunks: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in {
            "input_text",
            "output_text",
            "text",
        }:
            raise ValueError("only text input parts are supported by the local gateway")
        chunks.append(str(part.get("text", "")))
    return "".join(chunks)


def _tool_description(raw: dict[str, Any]) -> ToolDescription:
    if raw.get("type") != "function":
        raise ValueError("only function tools are supported")
    source = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    name = source.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("function tool requires a name")
    return ToolDescription.new(
        name=name,
        description=str(source.get("description", "")),
        parameters=source.get("parameters"),
    )


class HarmonyAdapter:
    """Official openai-harmony request renderer and completion parser."""

    def __init__(self, conversation_start_date: str | None = None) -> None:
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.conversation_start_date = conversation_start_date or date.today().isoformat()

    @property
    def stop_tokens(self) -> frozenset[int]:
        return frozenset(self.encoding.stop_tokens())

    def render_request(
        self,
        input_value: str | list[dict[str, Any]],
        *,
        instructions: str | None = None,
        tools: Sequence[dict[str, Any]] = (),
        reasoning_effort: str | None = None,
    ) -> list[int]:
        effort = {
            "low": ReasoningEffort.LOW,
            "medium": ReasoningEffort.MEDIUM,
            "high": ReasoningEffort.HIGH,
        }.get((reasoning_effort or "medium").lower())
        if effort is None:
            raise ValueError("reasoning effort must be low, medium, or high")

        messages = [
            Message(
                author=Author(role=Role.SYSTEM),
                content=[
                    SystemContent(
                        conversation_start_date=self.conversation_start_date,
                        reasoning_effort=effort,
                    )
                ],
            )
        ]
        developer_instructions = [instructions] if instructions else []
        input_messages: list[Message] = []
        function_names: dict[str, str] = {}
        raw_items = (
            [{"type": "message", "role": "user", "content": input_value}]
            if isinstance(input_value, str)
            else input_value
        )

        for item in raw_items:
            item_type = item.get("type", "message")
            if item_type == "function_call":
                name = str(item.get("name", ""))
                if not name:
                    raise ValueError("function_call input requires a name")
                if item.get("call_id"):
                    function_names[str(item["call_id"])] = name
                input_messages.append(
                    Message.from_role_and_content(Role.ASSISTANT, str(item.get("arguments", "{}")))
                    .with_channel("analysis")
                    .with_recipient(f"functions.{name}")
                    .with_content_type("json")
                )
                continue
            if item_type == "function_call_output":
                call_id = str(item.get("call_id", ""))
                name = str(item.get("name") or function_names.get(call_id) or call_id or "tool")
                if not name.startswith("functions."):
                    name = f"functions.{name}"
                input_messages.append(
                    Message.from_author_and_content(
                        Author(role=Role.TOOL, name=name), str(item.get("output", ""))
                    )
                )
                continue
            if item_type != "message":
                raise ValueError(f"unsupported response input item: {item_type}")
            role = Role(str(item.get("role", "user")))
            text = _text_content(item.get("content", ""))
            if role in {Role.SYSTEM, Role.DEVELOPER}:
                developer_instructions.append(text)
            else:
                message = Message.from_role_and_content(role, text)
                if role == Role.ASSISTANT:
                    message.with_channel("final")
                input_messages.append(message)

        tool_descriptions = [_tool_description(tool) for tool in tools]
        if developer_instructions or tool_descriptions:
            developer = DeveloperContent.new()
            if developer_instructions:
                developer.with_instructions("\n\n".join(developer_instructions))
            if tool_descriptions:
                developer.with_function_tools(tool_descriptions)
            messages.append(Message(author=Author(role=Role.DEVELOPER), content=[developer]))
        messages.extend(input_messages)
        conversation = Conversation.from_messages(messages)
        return self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

    def parse_completion(self, tokens: Sequence[int]) -> list[Message]:
        return self.encoding.parse_messages_from_completion_tokens(tokens, Role.ASSISTANT, strict=True)

    @staticmethod
    def message_text(message: Message) -> str:
        return "".join(content.text for content in message.content if isinstance(content, TextContent))

    def response_items(self, messages: Sequence[Message], response_id: str) -> list[dict[str, Any]]:
        suffix = response_id.removeprefix("resp_")
        output: list[dict[str, Any]] = []
        for message in messages:
            text = self.message_text(message)
            if message.recipient and message.recipient.startswith("functions."):
                index = len(output)
                output.append(
                    {
                        "id": f"fc_{suffix}_{index}",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": f"call_{suffix}_{index}",
                        "name": message.recipient.removeprefix("functions."),
                        "arguments": text,
                    }
                )
            elif message.channel in {"final", "commentary"}:
                index = len(output)
                output.append(
                    {
                        "id": f"msg_{suffix}_{index}",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": text,
                                "annotations": [],
                                "logprobs": [],
                            }
                        ],
                    }
                )
        return output
