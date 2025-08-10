import asyncio
import threading
import queue

from pathlib import Path
from typing import Any, Optional, Union
from pydantic import BaseModel, Field, create_model
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import chat

class OllamaMCP:

    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.initialized = threading.Event()
        self.tools: list[Any] = []
        self.thread = threading.Thread(target=self._run_background, daemon=True)
        self.thread.start()

    def _run_background(self):
        asyncio.run(self._async_run())

    async def _async_run(self):
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    print("Initializing MCP session...")
                    await session.initialize()
                    self.session = session
                    print("Listing available tools...")
                    tools_result = await session.list_tools()
                    self.tools = tools_result.tools
                    print(f"Found {len(self.tools)} tools: {[tool.name for tool in self.tools]}")
                    if not self.tools:
                        raise ValueError("No tools were found from the MCP server")
                    self.initialized.set()

                    while True:
                        try:
                            tool_name, arguments = self.request_queue.get(block=False)
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                            continue

                        if tool_name is None:
                            break
                        try:
                            result = await session.call_tool(tool_name, arguments)
                            self.response_queue.put(result)
                        except Exception as e:
                            self.response_queue.put(f"Error: {str(e)}")
        except Exception as e:
            print("MCP Session Initialization Error:", str(e))
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            self.initialized.set()  # Unblock waiting threads even if initialization failed.
            self.response_queue.put(f"MCP initialization error: {str(e)}")

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Post a tool call request and wait for a result.
        """
        if not self.initialized.wait(timeout=30):
            raise TimeoutError("MCP session did not initialize in time.")
        self.request_queue.put((tool_name, arguments))
        result = self.response_queue.get()
        return result

    def shutdown(self):
        """
        Cleanly shutdown the persistent session.
        """
        self.request_queue.put((None, None))
        self.thread.join()
        print("Persistent MCP session shut down.")


    @staticmethod
    def convert_json_type_to_python_type(json_type: str):
        """Simple mapping from JSON types to Python (Pydantic) types."""
        if json_type == "integer":
            return (int, ...)
        if json_type == "number":
            return (float, ...)
        if json_type == "string":
            return (str, ...)
        if json_type == "boolean":
            return (bool, ...)
        return (str, ...)

    def create_response_model(self):
        """
        Create a dynamic Pydantic response model based on the fetched tools.
        """
        dynamic_classes = {}
        for tool in self.tools:
            class_name = tool.name.capitalize()
            properties: dict[str, Any] = {}
            for prop_name, prop_info in tool.inputSchema.get("properties", {}).items():
                json_type = prop_info.get("type", "string")
                properties[prop_name] = self.convert_json_type_to_python_type(json_type)

            model = create_model(
                class_name,
                __base__=BaseModel,
                __doc__=tool.description,
                **properties,
            )
            dynamic_classes[class_name] = model

        if dynamic_classes:
            all_tools_type = Union[tuple(dynamic_classes.values())]
            Response = create_model(
                "Response",
                __base__=BaseModel,
                __doc__="LLm response class",
                response=(str, Field(..., description= "Confirmation to the user that the function will be called.")),
                tool=(all_tools_type, Field(
                    ...,
                    description="Tool to be used to run and get the magic outputs"
                )),
            )
        else:
            Response = create_model(
                "Response",
                __base__=BaseModel,
                __doc__="LLm response class",
                response=(str, ...),
                tool=(Optional[Any], Field(None, description="Tool to be used if not returning None")),
            )
        self.response_model = Response

    async def ollama_chat(self, messages: list[dict[str, str]]) -> Any:
        """
        Send messages to Ollama using the dynamic response model.
        If a tool is detected in the response, call it using the persistent session.
        """
        conversation = [{"role":"assistant", "content": f"you have  to use tools. You have the following functions at your disposal. {[ tool.name for tool in self.tools]}"}]
        conversation.extend(messages)
        if self.response_model is None:
            raise ValueError("Response model has not been created. Call create_response_model() first.")

        # Get the JSON schema for the chat message format.
        format_schema = self.response_model.model_json_schema()

        # Call Ollama (assumed synchronous) and parse the response.
        response = chat(
            model="wizardlm2:7b",
            messages=conversation,
            format=format_schema
        )
        print("Ollama response", response.message.content)
        response_obj = self.response_model.model_validate_json(response.message.content)
        maybe_tool = response_obj.tool

        if maybe_tool:
            # Handle both Pydantic model and dict cases
            if hasattr(maybe_tool, 'model_dump'):
                # It's a Pydantic model
                function_name = maybe_tool.__class__.__name__.lower()
                func_args = maybe_tool.model_dump()
            elif isinstance(maybe_tool, dict):
                # It's a dictionary, need to determine function name from available tools
                if self.tools:
                    function_name = self.tools[0].name  # Use first available tool
                    func_args = maybe_tool
                else:
                    raise ValueError("No tools available to call")
            else:
                raise ValueError(f"Unexpected tool type: {type(maybe_tool)}")
            
            # Use asyncio.to_thread to call the synchronous call_tool method in a thread.
            output = await asyncio.to_thread(self.call_tool, function_name, func_args)
            return output
        else:
            print("No tool detected in response. Returning plain response.")
        return response_obj.response


async def main():
    server_parameters = StdioServerParameters(
        command="python",
        args=["mcp-server-simple.py"],
        cwd=str(Path.cwd())
    )

    # Create the persistent session.
    persistent_session = OllamaMCP(server_parameters)

    # Wait until the session is fully initialized.
    if persistent_session.initialized.wait(timeout=30):
        print("Ready to call tools.")
        if not persistent_session.tools:
            print("Error: No tools available after initialization. Exiting.")
            persistent_session.shutdown()
            return
    else:
        print("Error: Initialization timed out.")
        persistent_session.shutdown()
        return

    # Create the dynamic response model from the retrieved tools.
    persistent_session.create_response_model()

    # Prepare messages for Ollama.

    messages = [
        {
            "role": "system",
            "content": (
                "You are an obedient assistant that has a list of tools in your context. "
                "Your task is to use this function to get the magic output. "
                "Do not generate the magic output yourself. "
                "Respond succinctly with a short message that references calling the function, "
                "but does not provide the function output itself. "
                "Return that short message in the 'response' property. "
                "For example: 'Sure, I'll run magicoutput function and return the output.' "
                "Also fill the 'tool' property with the correct arguments. "
            )
        },
        {
            "role": "user",
            "content": "Use the function to get the magic output for these parameters (obj1 = Wombat and obj2 = Dog)"
        }
    ]

    # Call Ollama and process the response.
    result = await persistent_session.ollama_chat(messages)
    print("Final result:", result)

    # Shutdown the persistent session once done.
    persistent_session.shutdown()

if __name__ == "__main__":
    asyncio.run(main())