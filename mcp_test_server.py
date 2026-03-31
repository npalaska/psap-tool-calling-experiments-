#!/usr/bin/env python3
"""
Test MCP (Model Context Protocol) Server
A simple HTTP-based MCP server for testing tool calling with vLLM

MCP Protocol: https://modelcontextprotocol.io/
"""

import json
import argparse
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """MCP Tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class MCPServerInfo:
    """MCP Server information"""
    name: str
    version: str
    protocolVersion: str = "2024-11-05"


class MCPTestServer:
    """
    Test MCP Server with various tools for validation
    """
    
    def __init__(self, server_name: str = "test-mcp-server"):
        self.server_info = MCPServerInfo(
            name=server_name,
            version="1.0.0"
        )
        self.tools = self._register_tools()
        self.call_history: List[Dict] = []
        
        # Mock data stores
        self.file_system: Dict[str, str] = {
            "/docs/readme.md": "# Welcome to the Test MCP Server\n\nThis is a mock file system.",
            "/docs/api.md": "# API Documentation\n\n## Endpoints\n- GET /tools\n- POST /call",
            "/data/config.json": '{"debug": true, "version": "1.0.0"}',
            "/data/users.json": '[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'
        }
        
        self.database: Dict[str, List[Dict]] = {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com", "role": "admin"},
                {"id": 2, "name": "Bob", "email": "bob@example.com", "role": "user"},
                {"id": 3, "name": "Charlie", "email": "charlie@example.com", "role": "user"}
            ],
            "products": [
                {"id": 1, "name": "Widget", "price": 9.99, "stock": 100},
                {"id": 2, "name": "Gadget", "price": 19.99, "stock": 50},
                {"id": 3, "name": "Gizmo", "price": 29.99, "stock": 25}
            ],
            "orders": [
                {"id": 1, "user_id": 1, "product_id": 1, "quantity": 2, "status": "completed"},
                {"id": 2, "user_id": 2, "product_id": 2, "quantity": 1, "status": "pending"}
            ]
        }
    
    def _register_tools(self) -> Dict[str, MCPTool]:
        """Register available tools"""
        tools = {
            "read_file": MCPTool(
                name="read_file",
                description="Read the contents of a file from the virtual filesystem",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    },
                    "required": ["path"]
                }
            ),
            "write_file": MCPTool(
                name="write_file",
                description="Write content to a file in the virtual filesystem",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }
            ),
            "list_files": MCPTool(
                name="list_files",
                description="List files in a directory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "The directory path to list",
                            "default": "/"
                        }
                    }
                }
            ),
            "query_database": MCPTool(
                name="query_database",
                description="Query the mock database. Supports tables: users, products, orders",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "The table to query (users, products, orders)",
                            "enum": ["users", "products", "orders"]
                        },
                        "filter": {
                            "type": "object",
                            "description": "Optional filter conditions as key-value pairs"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10
                        }
                    },
                    "required": ["table"]
                }
            ),
            "insert_record": MCPTool(
                name="insert_record",
                description="Insert a new record into a database table",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "The table to insert into",
                            "enum": ["users", "products", "orders"]
                        },
                        "record": {
                            "type": "object",
                            "description": "The record data to insert"
                        }
                    },
                    "required": ["table", "record"]
                }
            ),
            "execute_command": MCPTool(
                name="execute_command",
                description="Execute a simulated shell command (mock implementation)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute"
                        },
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Command arguments"
                        }
                    },
                    "required": ["command"]
                }
            ),
            "get_current_time": MCPTool(
                name="get_current_time",
                description="Get the current date and time",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Timezone (e.g., UTC, America/New_York)",
                            "default": "UTC"
                        },
                        "format": {
                            "type": "string",
                            "description": "Output format (iso, unix, human)",
                            "default": "iso"
                        }
                    }
                }
            ),
            "calculate": MCPTool(
                name="calculate",
                description="Perform mathematical calculations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate (e.g., '2 + 2 * 3')"
                        }
                    },
                    "required": ["expression"]
                }
            ),
            "search": MCPTool(
                name="search",
                description="Search across all data (files and database)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "scope": {
                            "type": "string",
                            "description": "Where to search (all, files, database)",
                            "enum": ["all", "files", "database"],
                            "default": "all"
                        }
                    },
                    "required": ["query"]
                }
            ),
            "echo": MCPTool(
                name="echo",
                description="Echo back the input (useful for testing)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to echo back"
                        }
                    },
                    "required": ["message"]
                }
            )
        }
        return tools
    
    def get_tools_list(self) -> List[Dict]:
        """Return list of tools in MCP format"""
        return [asdict(tool) for tool in self.tools.values()]
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result"""
        call_record = {
            "timestamp": datetime.now().isoformat(),
            "tool": name,
            "arguments": arguments
        }
        
        try:
            if name not in self.tools:
                raise ValueError(f"Unknown tool: {name}")
            
            # Dispatch to tool implementation
            result = self._execute_tool(name, arguments)
            call_record["success"] = True
            call_record["result"] = result
            
        except Exception as e:
            result = {"error": str(e), "type": type(e).__name__}
            call_record["success"] = False
            call_record["error"] = str(e)
        
        self.call_history.append(call_record)
        return result
    
    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute a specific tool"""
        
        if name == "read_file":
            path = args["path"]
            if path in self.file_system:
                return {"content": self.file_system[path], "path": path, "size": len(self.file_system[path])}
            else:
                raise FileNotFoundError(f"File not found: {path}")
        
        elif name == "write_file":
            path = args["path"]
            content = args["content"]
            self.file_system[path] = content
            return {"success": True, "path": path, "bytes_written": len(content)}
        
        elif name == "list_files":
            directory = args.get("directory", "/")
            files = [f for f in self.file_system.keys() if f.startswith(directory)]
            return {"directory": directory, "files": files, "count": len(files)}
        
        elif name == "query_database":
            table = args["table"]
            filter_cond = args.get("filter", {})
            limit = args.get("limit", 10)
            
            if table not in self.database:
                raise ValueError(f"Unknown table: {table}")
            
            results = self.database[table]
            
            # Apply filters
            if filter_cond:
                results = [
                    r for r in results 
                    if all(r.get(k) == v for k, v in filter_cond.items())
                ]
            
            return {"table": table, "results": results[:limit], "total": len(results)}
        
        elif name == "insert_record":
            table = args["table"]
            record = args["record"]
            
            if table not in self.database:
                raise ValueError(f"Unknown table: {table}")
            
            # Auto-generate ID
            max_id = max([r.get("id", 0) for r in self.database[table]], default=0)
            record["id"] = max_id + 1
            
            self.database[table].append(record)
            return {"success": True, "table": table, "record": record}
        
        elif name == "execute_command":
            command = args["command"]
            cmd_args = args.get("args", [])
            
            # Mock command execution
            mock_outputs = {
                "ls": f"file1.txt\nfile2.txt\ndir1/",
                "pwd": "/home/user",
                "whoami": "testuser",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "echo": " ".join(cmd_args)
            }
            
            output = mock_outputs.get(command, f"Command '{command}' executed (mock)")
            return {"command": command, "args": cmd_args, "output": output, "exit_code": 0}
        
        elif name == "get_current_time":
            fmt = args.get("format", "iso")
            now = datetime.now()
            
            if fmt == "iso":
                time_str = now.isoformat()
            elif fmt == "unix":
                time_str = str(int(now.timestamp()))
            else:
                time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
            
            return {"time": time_str, "format": fmt, "timezone": args.get("timezone", "UTC")}
        
        elif name == "calculate":
            expression = args["expression"]
            # Safe eval for basic math
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                raise ValueError("Invalid characters in expression")
            
            result = eval(expression)
            return {"expression": expression, "result": result}
        
        elif name == "search":
            query = args["query"].lower()
            scope = args.get("scope", "all")
            results = []
            
            if scope in ["all", "files"]:
                for path, content in self.file_system.items():
                    if query in content.lower() or query in path.lower():
                        results.append({"type": "file", "path": path, "match": query})
            
            if scope in ["all", "database"]:
                for table, records in self.database.items():
                    for record in records:
                        if any(query in str(v).lower() for v in record.values()):
                            results.append({"type": "database", "table": table, "record": record})
            
            return {"query": query, "scope": scope, "results": results, "count": len(results)}
        
        elif name == "echo":
            return {"message": args["message"], "echoed_at": datetime.now().isoformat()}
        
        else:
            raise ValueError(f"Tool not implemented: {name}")


class MCPHTTPHandler(BaseHTTPRequestHandler):
    """HTTP Handler for MCP requests"""
    
    server_instance: 'MCPHTTPServer' = None
    
    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")
    
    def _send_json_response(self, data: Dict, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def _send_error_response(self, message: str, status: int = 400):
        self._send_json_response({"error": message}, status)
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        mcp_server = self.server_instance.mcp_server
        
        if self.path == "/":
            self._send_json_response({
                "server": asdict(mcp_server.server_info),
                "endpoints": {
                    "GET /": "Server info",
                    "GET /tools": "List available tools",
                    "GET /history": "View call history",
                    "POST /call": "Call a tool"
                }
            })
        
        elif self.path == "/tools":
            self._send_json_response({
                "tools": mcp_server.get_tools_list()
            })
        
        elif self.path == "/history":
            self._send_json_response({
                "history": mcp_server.call_history[-50:]  # Last 50 calls
            })
        
        else:
            self._send_error_response(f"Unknown endpoint: {self.path}", 404)
    
    def do_POST(self):
        """Handle POST requests"""
        mcp_server = self.server_instance.mcp_server
        
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode()
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_error_response("Invalid JSON", 400)
            return
        
        if self.path == "/call":
            tool_name = data.get("name")
            arguments = data.get("arguments", {})
            
            if not tool_name:
                self._send_error_response("Missing 'name' field", 400)
                return
            
            result = mcp_server.call_tool(tool_name, arguments)
            
            # MCP response format
            self._send_json_response({
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result)
                    }
                ],
                "isError": "error" in result
            })
        
        elif self.path == "/rpc":
            # JSON-RPC style endpoint
            method = data.get("method")
            params = data.get("params", {})
            request_id = data.get("id")
            
            if method == "tools/list":
                result = {"tools": mcp_server.get_tools_list()}
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                result = mcp_server.call_tool(tool_name, arguments)
            else:
                self._send_json_response({
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": request_id
                })
                return
            
            self._send_json_response({
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            })
        
        else:
            self._send_error_response(f"Unknown endpoint: {self.path}", 404)


class MCPHTTPServer(HTTPServer):
    """HTTP Server with MCP support"""
    
    def __init__(self, address, handler_class, mcp_server: MCPTestServer):
        super().__init__(address, handler_class)
        self.mcp_server = mcp_server
        handler_class.server_instance = self


def main():
    parser = argparse.ArgumentParser(description="Test MCP Server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port to listen on (default: 8080)")
    parser.add_argument("--name", type=str, default="test-mcp-server",
                       help="Server name")
    
    args = parser.parse_args()
    
    mcp_server = MCPTestServer(server_name=args.name)
    http_server = MCPHTTPServer(
        (args.host, args.port),
        MCPHTTPHandler,
        mcp_server
    )
    
    print("="*70)
    print(f"MCP Test Server: {args.name}")
    print("="*70)
    print(f"\n  Listening on: http://{args.host}:{args.port}")
    print(f"  Tools available: {len(mcp_server.tools)}")
    print("\n  Endpoints:")
    print("    GET  /        - Server info")
    print("    GET  /tools   - List tools")
    print("    GET  /history - Call history")
    print("    POST /call    - Execute tool")
    print("    POST /rpc     - JSON-RPC endpoint")
    print("\n  Available tools:")
    for name, tool in mcp_server.tools.items():
        print(f"    - {name}: {tool.description[:50]}...")
    print("\n" + "="*70)
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        http_server.shutdown()


if __name__ == "__main__":
    main()
