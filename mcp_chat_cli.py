# mcp_chat_cli.py

import os
import asyncio
import json
import argparse
import sys
import signal
from pathlib import Path
from typing import Dict, Optional, Any, List
from enum import Enum
from contextlib import AsyncExitStack, suppress
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import litellm
from litellm import completion


import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ModelProvider(Enum):
    """Supported model providers."""
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"


class ModelNames:
    """Model identifiers for different providers."""
    CLAUDE3 = os.environ.get("CLAUDE3_MODEL_ID", "claude-3-7-sonnet-20250219")
    ANTHROPIC = f"anthropic/{CLAUDE3}"
    BEDROCK = os.environ.get(
        "BEDROCK_MODEL_ID", "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0")


class MCPToolProvider:
    """Manages MCP server connections and tool execution."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.environ.get(
            "MCP_CONFIG_PATH",
            str(Path.home() / ".mcp" / "config.json")
        )
        self.sessions = {}
        self.exit_stack = AsyncExitStack()
        self._shutdown_event = asyncio.Event()

    async def close(self):
        """Close all connections safely."""
        self._shutdown_event.set()
        try:
            async with asyncio.timeout(5.0):
                with suppress(asyncio.CancelledError):
                    await self.exit_stack.aclose()
        except asyncio.TimeoutError:
            print("MCP provider cleanup timed out")
        except Exception as e:
            print(f"Warning during MCP provider cleanup: {str(e)}")
        finally:
            self.sessions.clear()

    async def initialize(self):
        """Initialize all configured MCP servers."""
        await self.connect_servers()

    async def load_config(self) -> Dict:
        """Load the MCP server configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                return config.get("mcpServers", {})
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Invalid JSON in configuration file: {self.config_path}")
            return {}

    async def connect_servers(self) -> None:
        """Connect to all configured servers."""
        server_configs = await self.load_config()

        if not server_configs:
            print("No MCP servers configured.")
            return

        print(f"Found {len(server_configs)} MCP server(s)")

        for server_name, server_config in server_configs.items():
            try:
                await self.connect_server(server_name, server_config)
            except Exception as e:
                print(f"Failed to connect to server '{server_name}': {str(e)}")

    async def connect_server(self, server_name: str, server_config: Dict) -> None:
        """Connect to a specific MCP server."""
        command = server_config.get("command")
        args = server_config.get("args", [])
        env_vars = server_config.get("env", {})

        if not command:
            print(f"Missing 'command' for server '{server_name}'")
            return

        # Merge environment variables
        merged_env = os.environ.copy()
        for key, value in env_vars.items():
            merged_env[key] = value

        # Create server parameters
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=merged_env
        )

        try:
            # Create transport
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))

            # Create and initialize session
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio_transport[0], stdio_transport[1])
            )
            await session.initialize()

            # Store the session
            self.sessions[server_name] = session

            # Get server capabilities
            tools_result = await session.list_tools()

            if hasattr(tools_result, 'tools'):
                print(
                    f"Connected to '{server_name}' with {len(tools_result.tools)} tool(s)")
            else:
                print(
                    f"Connected to '{server_name}' but couldn't retrieve tools")

        except Exception as e:
            print(f"Error connecting to server '{server_name}': {str(e)}")
            raise

    async def get_available_tools(self) -> List[Dict]:
        """Get all available tools from connected servers in format for Anthropic API."""
        all_tools = []

        for server_name, session in self.sessions.items():
            try:
                tools_result = await session.list_tools()
                if hasattr(tools_result, 'tools'):
                    for tool in tools_result.tools:
                        tool_spec = {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description if hasattr(tool, "description") else "",
                                "parameters": tool.inputSchema if hasattr(tool, "inputSchema") else {}
                            }
                        }
                        # Store server name separately for internal use
                        tool_spec['_server'] = server_name
                        all_tools.append(tool_spec)
            except Exception as e:
                print(f"Error getting tools from '{server_name}': {str(e)}")

        return all_tools

    async def execute_tool(self, server_name: str, tool_name: str, tool_params: Dict) -> Any:
        """Execute a tool on a specific server."""
        session = self.sessions.get(server_name)
        if not session:
            raise ValueError(f"Server '{server_name}' not connected")

        try:
            result = await session.call_tool(tool_name, tool_params)
            return result
        except Exception as e:
            print(
                f"Error executing tool '{tool_name}' on server '{server_name}': {str(e)}")
            raise

    async def close(self):
        """Close all connections safely."""
        try:
            # Create a task in the current task group for cleanup
            async with asyncio.timeout(5.0):  # 5 second timeout for cleanup
                try:
                    await self.exit_stack.aclose()
                except (asyncio.CancelledError, Exception) as e:
                    print(f"Warning during MCP provider cleanup: {str(e)}")
        except asyncio.TimeoutError:
            print("MCP provider cleanup timed out")
        finally:
            self.sessions.clear()


class LLMProvider:
    """Provider for LLM services with a unified LiteLLM interface."""

    def __init__(self, provider_type: str = ModelProvider.ANTHROPIC.value):
        """Initialize the LLM provider.

        Args:
            provider_type: The type of provider to use (anthropic or bedrock)
        """
        self.provider_type = provider_type.lower()
        self.model = self._get_model_name()

        if self.provider_type == ModelProvider.BEDROCK.value:
            self._setup_bedrock()

    def _get_model_name(self) -> str:
        """Get the appropriate model name for the provider."""
        if self.provider_type == ModelProvider.ANTHROPIC.value:
            return ModelNames.ANTHROPIC
        elif self.provider_type == ModelProvider.BEDROCK.value:
            return ModelNames.BEDROCK
        raise ValueError(f"Unsupported provider type: {self.provider_type}")

    def _setup_bedrock(self) -> None:
        """Configure Bedrock client for LiteLLM."""
        import boto3
        session = boto3.Session(
            region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        litellm.aws_bedrock_client = session.client(
            service_name='bedrock-runtime'
        )

    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        """Generate a response using LiteLLM's unified interface.

        Args:
            messages: The conversation history
            tools: Optional list of available tools

        Returns:
            The LLM response
        """
        try:
            params = self._prepare_request_params(messages, tools)
            return await litellm.acompletion(**params)

        except Exception as e:
            print(f"Error calling LLM via LiteLLM: {e}")
            raise

    def _prepare_request_params(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Prepare the parameters for the LLM request.

        Args:
            messages: The conversation history
            tools: Optional list of available tools

        Returns:
            Dictionary of parameters for the LLM request
        """
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1000,
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        return params


class ChatApplication:
    """Main chat application integrating MCP tools with LLM."""

    def __init__(self, config_path: Optional[str] = None, model_provider: str = ModelProvider.ANTHROPIC.value):
        self.mcp_provider = MCPToolProvider(config_path)
        self.llm_provider = LLMProvider(model_provider)
        self.conversation_history = []
        self._shutdown_event = asyncio.Event()

    async def close(self):
        """Clean up resources safely."""
        self._shutdown_event.set()

        if hasattr(self, 'mcp_provider'):
            await self.mcp_provider.close()

        print("Application cleanup completed.")

    async def initialize(self):
        """Initialize the application."""
        await self.mcp_provider.initialize()

    async def process_query(self, query_text: str) -> str:
        """Process a user query using LLM and tools."""
        if self._shutdown_event.is_set():
            raise asyncio.CancelledError("Application is shutting down")
        try:
            # Start conversation with user query
            self._add_to_history("user", query_text)

            # Get available tools
            tools = await self.mcp_provider.get_available_tools()
            tool_to_server = {tool["function"]["name"]                              : tool["_server"] for tool in tools}

            # Get initial LLM response
            response = await self.llm_provider.generate_response(
                messages=self.conversation_history,
                tools=tools
            )

            result = await self._process_llm_response(response, tools, tool_to_server)
            self._add_to_history("assistant", result)

            return result

        except Exception as e:
            print(f"\nError processing query: {str(e)}")
            raise

    def _add_to_history(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content
        })

    async def _process_llm_response(
        self,
        response: Any,
        tools: List[Dict[str, Any]],
        tool_to_server: Dict[str, str]
    ) -> str:
        """Process the LLM response and handle any tool calls."""
        if not hasattr(response, 'choices') or not response.choices:
            return "I apologize, but I couldn't generate a proper response."

        choice = response.choices[0]
        message = choice.message

        # If no tool calls, return the content directly
        if not (hasattr(message, 'tool_calls') and message.tool_calls):
            return message.content or "I apologize, but I couldn't generate a proper response."

        # Process tool calls
        tool_calls = []
        for tool_call in message.tool_calls:
            tool_results = await self._execute_tool_call(
                tool_call,
                tool_to_server
            )
            tool_calls.extend(tool_results)

        # Get final response with tool results
        if tool_calls:
            full_messages = self.conversation_history + tool_calls
            final_response = await self.llm_provider.generate_response(
                messages=full_messages,
                tools=tools
            )

            if hasattr(final_response, 'choices') and final_response.choices:
                final_message = final_response.choices[0].message
                if hasattr(final_message, 'content'):
                    return final_message.content

        return "I apologize, but I couldn't process the tool results properly."

    async def _execute_tool_call(
        self,
        tool_call: Any,
        tool_to_server: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Execute a tool call and format the result for the conversation.

        Args:
            tool_call: The tool call from the LLM
            tool_to_server: Mapping of tool names to server names

        Returns:
            List of formatted messages for the conversation
        """
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool_id = tool_call.id

        print(f"🔧 Using tool: {tool_name}")

        server_name = tool_to_server.get(tool_name)
        if not server_name:
            raise ValueError(
                f"Could not determine which server provides tool '{tool_name}'")

        try:
            # Execute the tool
            tool_result = await self.mcp_provider.execute_tool(
                server_name,
                tool_name,
                tool_args
            )

            # Format the response messages
            return [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args)
                        }
                    }]
                },
                {
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_id
                }
            ]

        except Exception as e:
            return [{
                "role": "tool",
                "content": f"Error: {str(e)}",
                "tool_call_id": tool_id
            }]

# This is a duplicate method definition that should be removed completely


async def shutdown(app, signal=None):
    """Cleanup tasks tied to the service's shutdown."""
    if signal:
        print(f"\nReceived exit signal {signal.name}...")

    print("Initiating shutdown...")

    # Cancel all tasks first except cleanup
    tasks = [t for t in asyncio.all_tasks()
             if t is not asyncio.current_task()]

    if tasks:
        print(f"Cancelling {len(tasks)} outstanding tasks")
        for task in tasks:
            task.cancel()

        # Wait for tasks to complete their cancellation
        await asyncio.gather(*tasks, return_exceptions=True)

    # Perform simple cleanup
    await app.close()
    print("Shutdown complete.")


async def main_async():
    parser = argparse.ArgumentParser(
        description="MCP-powered chat application")
    parser.add_argument("--config", help="Path to MCP config file")
    parser.add_argument("--model-provider", default="anthropic",
                        choices=["anthropic", "bedrock"],
                        help="LLM provider (anthropic, bedrock)")
    args = parser.parse_args()

    app = ChatApplication(args.config, args.model_provider)

    def handle_signal(s):
        # Create task to run shutdown
        loop = asyncio.get_running_loop()
        loop.create_task(shutdown(app, signal=s))

    # Set up signal handlers
    loop = asyncio.get_running_loop()
    signals = (signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(s, lambda s=s: handle_signal(s))

    try:
        print("Initializing MCP servers...")
        await app.initialize()

        print("\n\U0001F916 Welcome to MCP Chat! Type 'exit' to quit.\n")

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    break

                print("\nProcessing...")
                response = await app.process_query(user_input)
                print(f"\nAI: {response}\n")
            except asyncio.CancelledError:
                print("\nOperation cancelled.")
                break
            except Exception as e:
                print(f"\nError processing query: {str(e)}")

    except asyncio.CancelledError:
        pass  # Handled by shutdown
    except Exception as e:
        print(f"Error: {str(e)}")


async def json_mode_main():
    """Run in JSON mode for the React Ink frontend."""
    # In your json_mode_main function (and other relevant functions)
    parser = argparse.ArgumentParser(
        description="MCP-powered chat application")
    parser.add_argument("--config", help="Path to MCP config file")
    parser.add_argument("--model-provider", default="anthropic",
                        choices=["anthropic", "bedrock"],
                        help="LLM provider (anthropic, bedrock)")
    parser.add_argument("--json-mode", action="store_true",
                        help="Run in JSON mode for frontend")
    args = parser.parse_args()

    # Create and initialize the application
    # Pass the config path
    app = ChatApplication(args.config, args.model_provider)

    try:
        # Initialize the application
        await app.initialize()

        # Signal initialization complete
        print(json.dumps({
            "type": "status",
            "status": "initialized",
            "servers": list(app.mcp_provider.sessions.keys())
        }))
        sys.stdout.flush()

        # Process input lines as JSON
        while True:
            line = sys.stdin.readline().strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                if "query" in request:
                    query = request["query"]

                    if query.lower() in ["exit", "quit"]:
                        break

                    # Process the query
                    response = await app.process_query(query)

                    # Return the response as JSON
                    print(json.dumps({
                        "type": "message",
                        "role": "assistant",
                        "content": response
                    }))
                    sys.stdout.flush()
            except json.JSONDecodeError:
                print(json.dumps({
                    "type": "error",
                    "error": "Invalid JSON"
                }))
                sys.stdout.flush()
            except Exception as e:
                print(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
                sys.stdout.flush()

    except Exception as e:
        print(json.dumps({
            "type": "error",
            "error": str(e)
        }))
        sys.stdout.flush()
    finally:
        pass


def main():
    """Entry point for the CLI app."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="MCP-powered chat application")
    parser.add_argument("--json-mode", action="store_true",
                        help="Run in JSON mode for frontend")
    args, _ = parser.parse_known_args()

    if args.json_mode:
        asyncio.run(json_mode_main())
    else:
        asyncio.run(main_async())


if __name__ == "__main__":
    main()
