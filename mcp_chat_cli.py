# mcp_chat_cli.py

import os
import asyncio
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List
from contextlib import AsyncExitStack
import json
import sys
import boto3
import json
from botocore.exceptions import ClientError


import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolProvider:
    """Manages MCP server connections and tool execution."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.environ.get(
            "MCP_CONFIG_PATH",
            str(Path.home() / ".mcp" / "config.json")
        )
        self.sessions = {}
        self.exit_stack = AsyncExitStack()

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
                            "name": tool.name,
                            "description": tool.description if hasattr(tool, "description") else "",
                            "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                            "server": server_name  # Add server name to identify where to execute
                        }
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
        """Close all connections."""
        await self.exit_stack.aclose()


class LLMProvider:
    """Provider for LLM services with a pluggable architecture."""

    def __init__(self, provider_type="anthropic"):
        self.provider_type = provider_type
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate LLM client based on provider type."""
        if self.provider_type == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set")
            return anthropic.Anthropic(api_key=api_key)
        elif self.provider_type == "bedrock":
            # Placeholder for AWS Bedrock integration
            return boto3.client(
                service_name="bedrock-runtime",
                region_name=os.environ.get("AWS_REGION", "us-east-1")
            )
        else:
            raise ValueError(
                f"Unsupported provider type: {self.provider_type}")

    async def generate_response(self, messages, tools=None, tool_choice=None):
        """Generate a response from the LLM."""
        if self.provider_type == "anthropic":
            # Use Anthropic's API
            kwargs = {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 1000,
                "messages": messages
            }

            if tools:
                # Format tools for Anthropic API
                anthropic_tools = []
                for tool in tools:
                    anthropic_tools.append({
                        "name": tool["name"],
                        "description": tool["description"],
                        "input_schema": tool["input_schema"]
                    })
                kwargs["tools"] = anthropic_tools

            return self.client.messages.create(**kwargs)
        elif self.provider_type == "bedrock":
            print("Using Bedrock provider")

            # Format messages for Claude on Bedrock
            formatted_messages = []
            for message in messages:
                if isinstance(message["content"], str):
                    formatted_messages.append({
                        "role": message["role"],
                        "content": [{"type": "text", "text": message["content"]}]
                    })
                else:
                    # Handle complex content objects
                    formatted_messages.append({
                        "role": message["role"],
                        "content": message["content"]
                    })

            print(
                f"Formatted messages for Bedrock: {json.dumps(formatted_messages, indent=2)}")

            # Prepare request payload
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": formatted_messages
            }

            # Add tools if provided
            if tools:
                anthropic_tools = []
                for tool in tools:
                    anthropic_tools.append({
                        "name": tool["name"],
                        "description": tool["description"],
                        "input_schema": {
                            "type": "object",
                            "properties": tool["input_schema"]["properties"] if "properties" in tool["input_schema"] else {},
                            "required": tool["input_schema"].get("required", [])
                        }
                    })
                request_body["tools"] = anthropic_tools
                print(f"Added {len(anthropic_tools)} tools to request")

            # Select the Anthropic model ID based on your preference
            model_id = os.environ.get(
                "BEDROCK_MODEL_ID", "anthropic.claude-3-7-sonnet-20250219-v1:0")
            print(f"Using Bedrock model ID: {model_id}")

            try:
                # Invoke the model
                print(
                    f"Sending request to Bedrock: {json.dumps(request_body, indent=2)}")
                response = self.client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body),

                )

                # Parse the response
                response_body = json.loads(
                    response["body"].read().decode("utf-8"))
                print(
                    f"Received response from Bedrock: {json.dumps(response_body, indent=2)}")

                # Define a class that simulates the Anthropic API response structure
                # Define a class that directly mimics the Anthropic API response structure
                class BedrockResponse:
                    class ContentItem:
                        def __init__(self, item_data):
                            self.type = item_data.get("type")
                            self.text = item_data.get("text", "")
                            self.id = item_data.get("id", "")
                            self.name = item_data.get("name", "")
                            self.input = item_data.get("input", {})

                        def __getitem__(self, key):
                            return getattr(self, key)

                    def __init__(self, bedrock_data):
                        self.model = model_id
                        self.stop_reason = bedrock_data.get("stop_reason")
                        self.content = []

                        # Process the content
                        print(f"Processing Bedrock response content")
                        if "content" in bedrock_data:
                            for i, item in enumerate(bedrock_data["content"]):
                                print(
                                    f"Content item {i}: {json.dumps(item, indent=2)}")
                                # Create ContentItem objects
                                self.content.append(self.ContentItem(item))
                                print(
                                    f"Added content item with type: {self.content[-1].type}")

                result = BedrockResponse(response_body)
                print(
                    f"Created BedrockResponse with {len(result.content)} content items")
                return result

            except ClientError as e:
                print(f"Error calling Bedrock: {e}")
                raise
            except Exception as e:
                print(f"Unexpected error with Bedrock: {e}")
                import traceback
                traceback.print_exc()
                raise

    def _format_complex_content(self, content):
        """Format complex content objects into strings for Bedrock."""
        if isinstance(content, list):
            formatted_parts = []
            for item in content:
                if item.get("type") == "text":
                    formatted_parts.append(item["text"])
                elif item.get("type") == "tool_result":
                    tool_result = f"Tool result for {item.get('tool_use_id', 'unknown tool')}:\n"
                    if isinstance(item.get("content"), list):
                        for content_part in item["content"]:
                            if content_part.get("type") == "text":
                                tool_result += content_part["text"]
                    else:
                        tool_result += str(item.get("content", ""))
                    formatted_parts.append(tool_result)
            return "\n".join(formatted_parts)
        return str(content)


class ChatApplication:
    """Main chat application integrating MCP tools with LLM."""

    def __init__(self, config_path=None, model_provider="anthropic"):
        self.mcp_provider = MCPToolProvider(config_path)
        self.llm_provider = LLMProvider(model_provider)
        self.conversation_history = []

    async def initialize(self):
        """Initialize the application."""
        await self.mcp_provider.initialize()

    async def process_query(self, query_text):
        """Process a user query using LLM and tools."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": query_text
        })

        # Get available tools
        tools = await self.mcp_provider.get_available_tools()

        # First LLM call to determine tool use
        response = await self.llm_provider.generate_response(
            messages=self.conversation_history,
            tools=tools
        )

        result = None
        tool_calls = []

        # Check if the model wants to use a tool
        for content in response.content:
            content_type = content.type if hasattr(
                content, 'type') else content.get('type')

            if content_type == 'text':
                result = content.text if hasattr(
                    content, 'text') else content.get('text')
            elif content_type == 'tool_use':
                # Track tool usage
                tool_name = content.name if hasattr(
                    content, 'name') else content.get('name')
                tool_args = content.input if hasattr(
                    content, 'input') else content.get('input')
                tool_id = content.id if hasattr(
                    content, 'id') else content.get('id')

                print(f"🔧 Using tool: {tool_name}")

                # Find which server this tool belongs to
                server_name = None
                for tool in tools:
                    if tool["name"] == tool_name:
                        server_name = tool["server"]
                        break

                if not server_name:
                    result = f"Error: Could not determine which server provides tool '{tool_name}'"
                    break

                try:
                    # Execute the tool
                    tool_result = await self.mcp_provider.execute_tool(server_name, tool_name, tool_args)

                    # Format tool result for the next LLM call
                    tool_calls.append({
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_args}]
                    })

                    if hasattr(tool_result, 'content'):
                        # Handle the case where content is a list of objects
                        if isinstance(tool_result.content, list):
                            # Convert to plain string if it's a list of content objects
                            tool_content = ""
                            for item in tool_result.content:
                                if hasattr(item, 'text'):
                                    tool_content += item.text
                                elif hasattr(item, 'type') and item.type == 'text':
                                    tool_content += item.text
                                else:
                                    tool_content += str(item)
                        else:
                            # Convert to plain string
                            tool_content = str(tool_result.content)
                    else:
                        tool_content = str(tool_result)

                    tool_calls.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": tool_content
                        }]
                    })
                except Exception as e:
                    tool_calls.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": f"Error: {str(e)}"
                        }]
                    })

        # If tools were used, make a follow-up call with the tool results
        if tool_calls:
            # Combine original conversation with tool calls
            full_messages = self.conversation_history + tool_calls

            # Make follow-up call
            final_response = await self.llm_provider.generate_response(
                messages=full_messages,
                tools=tools
            )

            # Update result with final response
            for content in final_response.content:
                content_type = content.type if hasattr(
                    content, 'type') else content.get('type')
                if content_type == 'text':
                    result = content.text if hasattr(
                        content, 'text') else content.get('text')
                    break

        # Add assistant's response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": result
        })

        return result

    async def close(self):
        """Clean up resources."""
        await self.mcp_provider.close()


async def main_async():
    parser = argparse.ArgumentParser(
        description="MCP-powered chat application")
    parser.add_argument("--config", help="Path to MCP config file")
    parser.add_argument("--model-provider", default="anthropic",
                        choices=["anthropic", "bedrock"],
                        help="LLM provider (anthropic, bedrock)")
    args = parser.parse_args()

    # Create and initialize the application
    app = ChatApplication(args.config, args.model_provider)

    try:
        print("Initializing MCP servers...")
        await app.initialize()

        print("\n🤖 Welcome to MCP Chat! Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            print("\nProcessing...")
            response = await app.process_query(user_input)
            print(f"\nAI: {response}\n")

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await app.close()


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
        await app.close()


def main():
    """Entry point for the CLI app."""
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
