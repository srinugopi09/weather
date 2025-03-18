import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.environ.get(
            "MCP_CONFIG_PATH",
            str(Path.home() / ".mcp" / "config.json")
        )
        self.servers = {}
        self.sessions = {}
        self.exit_stack = AsyncExitStack()

    async def load_config(self) -> Dict:
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
        server_configs = await self.load_config()

        if not server_configs:
            print("No servers configured. Please check your configuration file.")
            return

        print(f"Found {len(server_configs)} server(s) in configuration")

        for server_name, server_config in server_configs.items():
            try:
                await self.connect_server(server_name, server_config)
            except Exception as e:
                print(f"Failed to connect to server '{server_name}': {str(e)}")

    async def connect_server(self, server_name: str, server_config: Dict) -> None:
        """Connect to a specific server using an approach similar to Claude Desktop."""
        command = server_config.get("command")
        args = server_config.get("args", [])
        env_vars = server_config.get("env", {})

        if not command:
            print(f"Missing 'command' for server '{server_name}'")
            return

        print(
            f"Connecting to server '{server_name}' with command: {command} {' '.join(args)}")

        # Look for the command in PATH
        command_path = shutil.which(command)
        if command_path:
            print(f"Found command at: {command_path}")
            command = command_path
        else:
            print(f"Warning: Could not find '{command}' in PATH, using as is")

        # Merge current environment with provided env vars
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

            # Get server capabilities - properly handle the returned objects
            tools_result = await session.list_tools()
            resources_result = await session.list_resources()
            prompts_result = await session.list_prompts()

            print(f"Connected to server '{server_name}'")

            # Safely access tool counts
            if hasattr(tools_result, 'tools'):
                print(f"  Tools: {len(tools_result.tools)}")
                # Optionally show tool names
                for tool in tools_result.tools:
                    print(
                        f"    - {tool.name}: {tool.description if hasattr(tool, 'description') else ''}")
            else:
                print("  Tools: Unable to retrieve")

            # Safely access resource counts
            if hasattr(resources_result, 'resources'):
                print(f"  Resources: {len(resources_result.resources)}")
            else:
                print("  Resources: Unable to retrieve")

            # Safely access prompt counts
            if hasattr(prompts_result, 'prompts'):
                print(f"  Prompts: {len(prompts_result.prompts)}")
            else:
                print("  Prompts: Unable to retrieve")

            return session
        except Exception as e:
            print(f"Error connecting to server '{server_name}': {str(e)}")
            raise

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]):
        """Call a tool on a specific server."""
        session = self.sessions.get(server_name)
        if not session:
            raise ValueError(f"Server '{server_name}' not connected")

        print(
            f"Calling tool '{tool_name}' on server '{server_name}' with arguments: {arguments}")
        result = await session.call_tool(tool_name, arguments)
        return result

    async def read_resource(self, server_name: str, resource_uri: str):
        """Read a resource from a specific server."""
        session = self.sessions.get(server_name)
        if not session:
            raise ValueError(f"Server '{server_name}' not connected")

        print(f"Reading resource '{resource_uri}' from server '{server_name}'")
        content, mime_type = await session.read_resource(resource_uri)
        return content, mime_type

    async def get_prompt(self, server_name: str, prompt_name: str, arguments: Optional[Dict[str, str]] = None):
        """Get a prompt from a specific server."""
        session = self.sessions.get(server_name)
        if not session:
            raise ValueError(f"Server '{server_name}' not connected")

        print(
            f"Getting prompt '{prompt_name}' from server '{server_name}' with arguments: {arguments}")
        result = await session.get_prompt(prompt_name, arguments)
        return result

    async def close(self):
        """Close all connections."""
        try:
            await self.exit_stack.aclose()
            print("All server connections closed")
        except Exception as e:
            print(f"Error closing connections: {str(e)}")


async def main():
    client = MCPClient("./mcp_config.json")

    try:
        await client.connect_servers()

        # Example: interact with connected servers
        connected_servers = list(client.sessions.keys())

        if not connected_servers:
            print("No servers connected. Exiting.")
            return

        print("\nConnected servers:", connected_servers)

        # Example for weather server
        if "weather" in client.sessions:
            print("\nTesting weather server...")
            try:
                result = await client.call_tool(
                    "weather",
                    "get_forecast",
                    {"latitude": 37.7749, "longitude": -122.4194}
                )
                print(f"Weather forecast result: {result}")

                result = await client.call_tool(
                    "weather",
                    "get_alerts",
                    {"state": "CA"}
                )
                print(f"Weather alerts result: {result}")
            except Exception as e:
                print(f"Error calling weather tools: {str(e)}")

        print("\nClient running. Press Ctrl+C to exit.")
        await asyncio.sleep(60)  # Run for a minute

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
