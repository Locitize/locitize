"""Test-only MCP server: one fixed fixture read, no caller-selected paths."""
from pathlib import Path
import sys


def read_fixed_fixture(path: Path) -> str:
    if path.name != "perimeter.py" or path.parent.name != "workspace" or not path.is_file() or path.is_symlink():
        raise ValueError("Invalid test fixture")
    return path.read_text(encoding="ascii")


def main() -> None:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    path = Path(sys.argv[1]).resolve(strict=True)
    # The trusted harness supplies this fixed path at startup; tools take no path.
    server = FastMCP("LOCITIZE fixed fixture reader")

    # Truthful hints: one fixed local read. The client's policy decides approval.
    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                                             idempotentHint=True, openWorldHint=False))
    def read_fixture() -> str:
        """Read the fixed Python test fixture without changing anything."""
        return read_fixed_fixture(path)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
