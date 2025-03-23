# Databricks notebook source
import json
import time
from collections.abc import Generator
from typing import Any, cast

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model.llm import LLMModelConfig, LLMResult, LLMResultChunk
from dify_plugin.entities.model.message import (
    PromptMessageTool,
    UserPromptMessage,
)
from dify_plugin.entities.tool import ToolInvokeMessage, ToolParameter, ToolProviderType
from dify_plugin.interfaces.agent import AgentModelConfig, AgentStrategy, ToolEntity
from pydantic import BaseModel

class BasicParams(BaseModel):
    maximum_iterations: int
    model: AgentModelConfig
    tools: list[ToolEntity]
    query: str

class NgsAgentAgentStrategy(AgentStrategy):
    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage]:
        params = BasicParams(**parameters)
        function_call_round_log = self.create_log_message(
            label="Function Call Round1 ",
            data={},
            metadata={},
        )
        yield function_call_round_log
        model_started_at = time.perf_counter()
        model_log = self.create_log_message(
            label=f"{params.model.model} Thought",
            data={},
            metadata={"start_at": model_started_at, "provider": params.model.provider},
            status=ToolInvokeMessage.LogMessage.LogStatus.START,
            parent=function_call_round_log,
        )
        yield model_log
        chunks: Generator[LLMResultChunk, None, None] | LLMResult = (
            self.session.model.llm.invoke(
                model_config=LLMModelConfig(**params.model.model_dump(mode="json")),
                prompt_messages=[UserPromptMessage(content=params.query)],
                tools=[
                    self._convert_tool_to_prompt_message_tool(tool)
                    for tool in params.tools
                ],
                stop=params.model.completion_params.get("stop", [])
                if params.model.completion_params
                else [],
                stream=True,
            )
        )
        response = ""
        tool_calls = []
        tool_instances = (
            {tool.identity.name: tool for tool in params.tools} if params.tools else {}
        )
        tool_call_names = ""
        tool_call_inputs = ""
        for chunk in chunks:
            # check if there is any tool call
            if self.check_tool_calls(chunk):
                tool_calls = self.extract_tool_calls(chunk)
                tool_call_names = ";".join([tool_call[1] for tool_call in tool_calls])
                try:
                    tool_call_inputs = json.dumps(
                        {tool_call[1]: tool_call[2] for tool_call in tool_calls},
                        ensure_ascii=False,
                    )
                except json.JSONDecodeError:
                    # ensure ascii to avoid encoding error
                    tool_call_inputs = json.dumps(
                        {tool_call[1]: tool_call[2] for tool_call in tool_calls}
                    )
                print(tool_call_names, tool_call_inputs)
            if chunk.delta.message and chunk.delta.message.content:
                if isinstance(chunk.delta.message.content, list):
                    for content in chunk.delta.message.content:
                        response += content.data
                        print(content.data, end="", flush=True)
                else:
                    response += str(chunk.delta.message.content)
                    print(str(chunk.delta.message.content), end="", flush=True)

            if chunk.delta.usage:
                # usage of the model
                usage = chunk.delta.usage

        yield self.finish_log_message(
            log=model_log,
            data={
                "output": response,
                "tool_name": tool_call_names,
                "tool_input": tool_call_inputs,
            },
            metadata={
                "started_at": model_started_at,
                "finished_at": time.perf_counter(),
                "elapsed_time": time.perf_counter() - model_started_at,
                "provider": params.model.provider,
            },
        )
        yield self.create_text_message(
            text=f"{response or json.dumps(tool_calls, ensure_ascii=False)}\n"
        )
        result = ""
        for tool_call_id, tool_call_name, tool_call_args in tool_calls:
            tool_instance = tool_instances[tool_call_name]
            tool_invoke_responses = self.session.tool.invoke(
                provider_type=ToolProviderType.BUILT_IN,
                provider=tool_instance.identity.provider,
                tool_name=tool_instance.identity.name,
                parameters={**tool_instance.runtime_parameters, **tool_call_args},
            )
            if not tool_instance:
                tool_invoke_responses = {
                    "tool_call_id": tool_call_id,
                    "tool_call_name": tool_call_name,
                    "tool_response": f"there is not a tool named {tool_call_name}",
                }
            else:
                # invoke tool
                tool_invoke_responses = self.session.tool.invoke(
                    provider_type=ToolProviderType.BUILT_IN,
                    provider=tool_instance.identity.provider,
                    tool_name=tool_instance.identity.name,
                    parameters={**tool_instance.runtime_parameters, **tool_call_args},
                )
                result = ""
                for tool_invoke_response in tool_invoke_responses:
                    if tool_invoke_response.type == ToolInvokeMessage.MessageType.TEXT:
                        result += cast(
                            ToolInvokeMessage.TextMessage, tool_invoke_response.message
                        ).text
                    elif (
                        tool_invoke_response.type == ToolInvokeMessage.MessageType.LINK
                    ):
                        result += (
                            f"result link: {cast(ToolInvokeMessage.TextMessage, tool_invoke_response.message).text}."
                            + " please tell user to check it."
                        )
                    elif tool_invoke_response.type in {
                        ToolInvokeMessage.MessageType.IMAGE_LINK,
                        ToolInvokeMessage.MessageType.IMAGE,
                    }:
                        result += (
                            "image has been created and sent to user already, "
                            + "you do not need to create it, just tell the user to check it now."
                        )
                    elif (
                        tool_invoke_response.type == ToolInvokeMessage.MessageType.JSON
                    ):
                        text = json.dumps(
                            cast(
                                ToolInvokeMessage.JsonMessage,
                                tool_invoke_response.message,
                            ).json_object,
                            ensure_ascii=False,
                        )
                        result += f"tool response: {text}."
                    else:
                        result += f"tool response: {tool_invoke_response.message!r}."

                tool_response = {
                    "tool_call_id": tool_call_id,
                    "tool_call_name": tool_call_name,
                    "tool_response": result,
                }
        yield self.create_text_message(result)

    def _convert_tool_to_prompt_message_tool(
        self, tool: ToolEntity
    ) -> PromptMessageTool:
        """
        convert tool to prompt message tool
        """
        message_tool = PromptMessageTool(
            name=tool.identity.name,
            description=tool.description.llm if tool.description else "",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

        parameters = tool.parameters
        for parameter in parameters:
            if parameter.form != ToolParameter.ToolParameterForm.LLM:
                continue

            parameter_type = parameter.type
            if parameter.type in {
                ToolParameter.ToolParameterType.FILE,
                ToolParameter.ToolParameterType.FILES,
            }:
                continue
            enum = []
            if parameter.type == ToolParameter.ToolParameterType.SELECT:
                enum = (
                    [option.value for option in parameter.options]
                    if parameter.options
                    else []
                )

            message_tool.parameters["properties"][parameter.name] = {
                "type": parameter_type,
                "description": parameter.llm_description or "",
            }

            if len(enum) > 0:
                message_tool.parameters["properties"][parameter.name]["enum"] = enum

            if parameter.required:
                message_tool.parameters["required"].append(parameter.name)

        return message_tool

    def check_tool_calls(self, llm_result_chunk: LLMResultChunk) -> bool:
        """
        Check if there is any tool call in llm result chunk
        """
        return bool(llm_result_chunk.delta.message.tool_calls)

    def extract_tool_calls(
        self, llm_result_chunk: LLMResultChunk
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """
        Extract tool calls from llm result chunk

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: [(tool_call_id, tool_call_name, tool_call_args)]
        """
        tool_calls = []
        for prompt_message in llm_result_chunk.delta.message.tool_calls:
            args = {}
            if prompt_message.function.arguments != "":
                args = json.loads(prompt_message.function.arguments)

            tool_calls.append(
                (
                    prompt_message.id,
                    prompt_message.function.name,
                    args,
                )
            )

        return tool_calls