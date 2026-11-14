---
name: reactbits
description: >-
  Browse, search, and fetch animated backgrounds, text effects, and interactive UI
  components from the React Bits (@react-bits) registry (https://reactbits.dev/r/{name}.json).
---

# React Bits MCP & Registry Skill

This workspace is configured with the `@react-bits` registry in [`components.json`](../../components.json) (`https://reactbits.dev/r/{name}.json`) and the local MCP server in [`scripts/reactbits_mcp_server.py`](../../scripts/reactbits_mcp_server.py).

## Quick Commands

1. **List all available React Bits components** (or filter by `Backgrounds`, `Text Animations`, `Animations`, `Components`):
   ```bash
   python3 scripts/reactbits_mcp_server.py --list
   python3 scripts/reactbits_mcp_server.py --list --category "Backgrounds"
   ```

2. **Search components by effect or use case**:
   ```bash
   python3 scripts/reactbits_mcp_server.py --search "aurora"
   ```

3. **Fetch live component code from `https://reactbits.dev/r/{name}-{variant}.json`** (`JS-CSS`, `JS-TW`, `TS-CSS`, `TS-TW`):
   ```bash
   python3 scripts/reactbits_mcp_server.py --fetch Aurora --variant JS-CSS
   python3 scripts/reactbits_mcp_server.py --fetch SplitText --variant JS-CSS
   python3 scripts/reactbits_mcp_server.py --fetch SpotlightCard --variant JS-CSS
   ```
