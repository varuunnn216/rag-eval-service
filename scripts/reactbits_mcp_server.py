#!/usr/bin/env python3
"""
React Bits MCP Server (`scripts/reactbits_mcp_server.py`).

Implements the Model Context Protocol (MCP, JSON-RPC 2.0 over stdio) connected to the
official `@react-bits` registry configured in `components.json`
(`https://reactbits.dev/r/{name}.json`).

Works out-of-the-box with system Python 3 without requiring Node.js / npx, and can
also be invoked directly from the command line:
    python3 scripts/reactbits_mcp_server.py --list
    python3 scripts/reactbits_mcp_server.py --search "aurora"
    python3 scripts/reactbits_mcp_server.py --fetch Aurora --variant JS-CSS
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPONENTS_JSON_PATH = PROJECT_ROOT / "components.json"

DEFAULT_REGISTRY_TEMPLATE = "https://reactbits.dev/r/{name}.json"
VALID_VARIANTS = ("JS-CSS", "JS-TW", "TS-CSS", "TS-TW")

REACTBITS_CATALOG: List[Dict[str, str]] = [
    # Backgrounds
    {
        "name": "GradientASCII",
        "category": "Backgrounds",
        "description": "Dynamic flowing multi-stop color gradient mapped through an ASCII character density ramp with mouse-interaction scrambling and click shockwave ripples.",
        "best_for": "Cyber-technical hero backdrop and full-viewport ambient ASCII shader",
    },
    {
        "name": "Aurora",
        "category": "Backgrounds",
        "description": "Flowing northern-lights WebGL shader backdrop with customizable color stops, amplitude, and blend.",
        "best_for": "Hero section backdrop, AI product home page header",
    },
    {
        "name": "SoftAurora",
        "category": "Backgrounds",
        "description": "Diffused ambient aurora glow with smooth organic motion.",
        "best_for": "Subtle dark-mode hero background",
    },
    {
        "name": "Beams",
        "category": "Backgrounds",
        "description": "Animated volumetric light beams radiating across a dark canvas.",
        "best_for": "Landing page hero spotlight",
    },
    {
        "name": "Dither",
        "category": "Backgrounds",
        "description": "Retro Bayer-matrix dithered shader wave background with interactive mouse ripple.",
        "best_for": "Developer tool / technical workbench hero",
    },
    {
        "name": "Particles",
        "category": "Backgrounds",
        "description": "Interactive floating 3D/2D particle field that reacts to cursor movement.",
        "best_for": "Knowledge graph / vector search visualization backdrop",
    },
    {
        "name": "Hyperspeed",
        "category": "Backgrounds",
        "description": "High-speed 3D light-trail warp tunnel with customizable road/lane colors.",
        "best_for": "High-energy showcase hero",
    },
    {
        "name": "Threads",
        "category": "Backgrounds",
        "description": "Flowing interactive harmonic wave threads that bend toward the cursor.",
        "best_for": "Modern AI / RAG pipeline hero section",
    },
    {
        "name": "Orb",
        "category": "Backgrounds",
        "description": "Luminous morphing energy orb shader with hover distortion.",
        "best_for": "AI assistant centerpiece or search callout",
    },
    {
        "name": "GridMotion",
        "category": "Backgrounds",
        "description": "Dynamic perspective grid with smooth kinetic motion.",
        "best_for": "Feature showcase backdrop",
    },
    {
        "name": "Squares",
        "category": "Backgrounds",
        "description": "Animated scrolling interactive grid of squares with hover highlight.",
        "best_for": "Clean technical dashboard background",
    },
    {
        "name": "Iridescence",
        "category": "Backgrounds",
        "description": "Holographic liquid iridescent surface shader.",
        "best_for": "Accent cards and hero banners",
    },
    {
        "name": "LiquidChrome",
        "category": "Backgrounds",
        "description": "Reflective metallic liquid shader with interactive ripples.",
        "best_for": "Futuristic landing page hero",
    },
    {
        "name": "Waves",
        "category": "Backgrounds",
        "description": "Interactive Perlin-noise vector wave lines responding to pointer velocity.",
        "best_for": "Full-page ambient background",
    },
    {
        "name": "Prism",
        "category": "Backgrounds",
        "description": "Refractive 3D prism light dispersion shader.",
        "best_for": "Hero header visual",
    },
    # Text Animations
    {
        "name": "ASCIIText",
        "category": "Text Animations",
        "description": "Interactive 3D-tilting hero and section headings constructed entirely from ASCII density glyphs with cursor tilt, hover scramble, click ripple, and chromatic color fringe.",
        "best_for": "Hero banner centerpiece title and interactive section headers",
    },
    {
        "name": "SplitText",
        "category": "Text Animations",
        "description": "Staggered per-character or per-word entrance animation with spring physics.",
        "best_for": "Main hero headline entrance",
    },
    {
        "name": "BlurText",
        "category": "Text Animations",
        "description": "Smooth blur-to-sharp staggered word/letter reveal on load or scroll.",
        "best_for": "Hero title and section headers",
    },
    {
        "name": "ShinyText",
        "category": "Text Animations",
        "description": "Metallic light-sweep shimmer across badge or heading text.",
        "best_for": "Hero announcement pill, status badges, CTA labels",
    },
    {
        "name": "GradientText",
        "category": "Text Animations",
        "description": "Animated multi-stop flowing gradient text with optional glowing border.",
        "best_for": "Highlighted keywords in hero headline",
    },
    {
        "name": "DecryptedText",
        "category": "Text Animations",
        "description": "Cipher/matrix character scramble that decodes into the target string on view or hover.",
        "best_for": "Technical metrics, chunk IDs, or developer hero subheadings",
    },
    {
        "name": "RotatingText",
        "category": "Text Animations",
        "description": "Smooth vertical/3D flip ticker cycling through multiple phrases.",
        "best_for": "Dynamic hero value proposition ('Hybrid RRF', 'Cross-Encoder Rerank', 'Verified Citations')",
    },
    {
        "name": "CountUp",
        "category": "Text Animations",
        "description": "Spring-animated numeric counter from start to target value.",
        "best_for": "KPI metrics cards, latency & recall scorecards",
    },
    {
        "name": "TrueFocus",
        "category": "Text Animations",
        "description": "Cycles a sharp focus bracket across words while blurring inactive words.",
        "best_for": "Feature callouts and pipeline stage highlights",
    },
    {
        "name": "ScrollReveal",
        "category": "Text Animations",
        "description": "Progressive word opacity & unblur reveal linked to scroll position.",
        "best_for": "Architecture overview section",
    },
    # Animations & Micro-Interactions
    {
        "name": "FadeContent",
        "category": "Animations",
        "description": "Configurable fade + blur + translateY entrance wrapper on viewport entry.",
        "best_for": "Section cards and staggered dashboard panels",
    },
    {
        "name": "AnimatedContent",
        "category": "Animations",
        "description": "Spring-driven slide/scale container entrance animation.",
        "best_for": "Feature cards, query result panels",
    },
    {
        "name": "StarBorder",
        "category": "Animations",
        "description": "Animated glowing star-trail border orbiting around a button or card.",
        "best_for": "Primary 'Run' CTA button or featured card",
    },
    {
        "name": "Magnet",
        "category": "Animations",
        "description": "Magnetic cursor-pull wrapper for buttons and interactive chips.",
        "best_for": "CTA buttons and sample query chips",
    },
    {
        "name": "ClickSpark",
        "category": "Animations",
        "description": "Radiating particle spark burst on click anywhere inside the wrapper.",
        "best_for": "Interactive buttons and preset chips",
    },
    {
        "name": "GlareHover",
        "category": "Animations",
        "description": "Realistic diagonal specular light sheen across a card on hover.",
        "best_for": "KPI cards and retrieved chunk cards",
    },
    # Components
    {
        "name": "SpotlightCard",
        "category": "Components",
        "description": "Card surface with a radial cursor-tracking spotlight glow.",
        "best_for": "Feature grid cards, citation cards, and metric boxes",
    },
    {
        "name": "TiltedCard",
        "category": "Components",
        "description": "3D perspective tilt card that tracks mouse coordinates with spring damping.",
        "best_for": "Hero showcase cards and interactive architecture tiles",
    },
    {
        "name": "Dock",
        "category": "Components",
        "description": "macOS-style magnification dock with smooth Gaussian neighbor scaling.",
        "best_for": "Floating navigation bar or quick-action switcher",
    },
    {
        "name": "FluidGlass",
        "category": "Components",
        "description": "Frosted glassmorphic container with subtle chromatic refraction.",
        "best_for": "Hero overlay bar and floating search box",
    },
    {
        "name": "PixelCard",
        "category": "Components",
        "description": "Card that shimmers with animated canvas pixels on hover.",
        "best_for": "Interactive feature highlights",
    },
]


def get_registry_template() -> str:
    """Read `@react-bits` registry URL template from `components.json`."""
    if COMPONENTS_JSON_PATH.exists():
        try:
            data = json.loads(COMPONENTS_JSON_PATH.read_text(encoding="utf-8"))
            registries = data.get("registries") or {}
            if "@react-bits" in registries:
                return str(registries["@react-bits"])
        except Exception:
            pass
    return DEFAULT_REGISTRY_TEMPLATE


def list_components(category: Optional[str] = None) -> List[Dict[str, str]]:
    if not category or category.lower() == "all":
        return REACTBITS_CATALOG
    cat_lower = category.strip().lower()
    return [c for c in REACTBITS_CATALOG if cat_lower in c["category"].lower()]


def search_components(query: str) -> List[Dict[str, str]]:
    q = (query or "").strip().lower()
    if not q:
        return REACTBITS_CATALOG
    results = []
    for item in REACTBITS_CATALOG:
        hay = f"{item['name']} {item['category']} {item['description']} {item['best_for']}".lower()
        if q in hay:
            results.append(item)
    return results


def fetch_component_registry_json(name: str, variant: str = "JS-CSS") -> Dict[str, Any]:
    """Fetch component definition from `https://reactbits.dev/r/{name}-{variant}.json`."""
    clean_name = name.strip()
    clean_variant = variant.strip().upper() if variant else "JS-CSS"
    if clean_variant not in VALID_VARIANTS:
        clean_variant = "JS-CSS"

    # If the caller already passed e.g. 'Aurora-JS-CSS', keep it as-is
    if any(clean_name.endswith(f"-{v}") for v in VALID_VARIANTS):
        registry_slug = clean_name
    else:
        registry_slug = f"{clean_name}-{clean_variant}"

    template = get_registry_template()
    url = template.replace("{name}", registry_slug)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ReactBits-MCP-Client/1.0",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return {
            "status": "ok",
            "registry_url": url,
            "slug": registry_slug,
            "component": payload,
        }
    except urllib.error.HTTPError as exc:
        return {
            "status": "error",
            "registry_url": url,
            "slug": registry_slug,
            "error": f"HTTP {exc.code}: {exc.reason}",
        }
    except Exception as exc:
        return {
            "status": "error",
            "registry_url": url,
            "slug": registry_slug,
            "error": str(exc),
        }


# =====================================================================
# JSON-RPC 2.0 STDIO MCP SERVER
# =====================================================================
MCP_TOOLS = [
    {
        "name": "list_reactbits_components",
        "description": (
            "List available React Bits components (Backgrounds, Text Animations, "
            "Animations, Components) with descriptions and recommended home page use cases."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category filter: 'Backgrounds', 'Text Animations', 'Animations', 'Components', or 'all'.",
                }
            },
        },
    },
    {
        "name": "search_reactbits_components",
        "description": "Search the React Bits catalog by keyword, effect, or use case.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keyword (e.g. 'aurora', 'spotlight', 'text', 'hero', 'card').",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_reactbits_component",
        "description": (
            "Fetch the official React Bits component code and dependencies from "
            "https://reactbits.dev/r/{name}-{variant}.json."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Component name in PascalCase (e.g. 'Aurora', 'SplitText', 'SpotlightCard', 'ShinyText', 'DecryptedText').",
                },
                "variant": {
                    "type": "string",
                    "enum": ["JS-CSS", "JS-TW", "TS-CSS", "TS-TW"],
                    "description": "Code variant (defaults to 'JS-CSS').",
                },
            },
            "required": ["name"],
        },
    },
]


def _handle_rpc_request(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "reactbits-mcp", "version": "1.0.0"},
            },
        }

    if method.startswith("notifications/"):
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": MCP_TOOLS},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments") or {}

        if tool_name == "list_reactbits_components":
            items = list_components(args.get("category"))
            text = json.dumps(items, indent=2)
        elif tool_name == "search_reactbits_components":
            items = search_components(args.get("query", ""))
            text = json.dumps(items, indent=2)
        elif tool_name == "get_reactbits_component":
            data = fetch_component_registry_json(
                name=args.get("name", "Aurora"),
                variant=args.get("variant", "JS-CSS"),
            )
            text = json.dumps(data, indent=2)
        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": False,
            },
        }

    if msg_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {},
        }
    return None


def run_stdio_mcp_server() -> None:
    """Run the JSON-RPC 2.0 MCP loop over stdin/stdout."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            resp = _handle_rpc_request(msg)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except Exception as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(exc)},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="React Bits MCP Server & Registry Client")
    parser.add_argument("--list", action="store_true", help="List catalog of React Bits components")
    parser.add_argument("--category", default=None, help="Filter catalog by category")
    parser.add_argument("--search", default=None, help="Search React Bits catalog by keyword")
    parser.add_argument("--fetch", default=None, help="Fetch component JSON from reactbits.dev/r/{name}.json")
    parser.add_argument("--variant", default="JS-CSS", help="Variant: JS-CSS, JS-TW, TS-CSS, TS-TW")
    args = parser.parse_args()

    if args.list:
        print(json.dumps(list_components(args.category), indent=2))
        return
    if args.search:
        print(json.dumps(search_components(args.search), indent=2))
        return
    if args.fetch:
        print(json.dumps(fetch_component_registry_json(args.fetch, args.variant), indent=2))
        return

    run_stdio_mcp_server()


if __name__ == "__main__":
    main()
