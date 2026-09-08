"""Config walkthrough: load and inspect AgentAblit configuration programmatically.

Demonstrates how the YAML config maps to the ProxyConfig dataclass, and how
environment variables override file values.

Prerequisites:
  pip install pyyaml
"""
import sys
from pathlib import Path

# Add src to path for imports
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def walk_config():
    """Load and display the current configuration structure."""
    try:
        from proxy.config import ProxyConfig
    except ImportError:
        print("Could not import ProxyConfig. Ensure you're running from the project root.")
        print("  cd src && python ../examples/config_walkthrough.py")
        return

    # Load from config.yaml (or env vars)
    cfg = ProxyConfig.load()

    print("=== AgentAblit Configuration ===\n")

    print("Host (upstream model):")
    print(f"  URL:   {_mask(cfg.host_url)}")
    print(f"  Model: {cfg.host_model}")
    print(f"  Timeout: {cfg.host_timeout}s")

    print("\nParasite/B (continuation model):")
    print(f"  URL:   {_mask(cfg.parasite_url)}")
    print(f"  Model: {cfg.parasite_model}")
    print(f"  Timeout: {cfg.parasite_timeout}s")

    print("\nMechanism:")
    print(f"  Version: {cfg.mechanism_version}")
    print(f"  Ablate L3: {cfg.ablate_l3}")

    print("\nEngine selectors (execution profiles):")
    print(f"  Full:        recover + reconstruct + L3 escalation")
    print(f"  Recover:     recover only (no reconstruct)")
    print(f"  Passthrough: format-conversion relay only")

    print("\nConfig file locations (in priority order):")
    print("  1. Environment variables (ABLIT_*)")
    print("  2. config.yaml in current directory")
    print("  3. config.example.yaml (defaults)")


def _mask(url: str) -> str:
    """Mask sensitive parts of a URL for display."""
    if not url:
        return "(not set)"
    # Just show the host:port, hide path and credentials
    from urllib.parse import urlsplit
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or "?"
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{host}{port}/..."
    except Exception:
        return "(invalid URL)"


if __name__ == "__main__":
    walk_config()
