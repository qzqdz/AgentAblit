# AgentAblit Examples

Integration examples and quick-start demos for AgentAblit.

## Examples

| File | What it shows |
| :--- | :--- |
| [`minimal_client.py`](minimal_client.py) | Point an OpenAI-compatible client at the relay |
| [`config_walkthrough.py`](config_walkthrough.py) | Programmatic config loading and inspection |

## Quick Start

1. Start the relay:
   ```sh
   cd ../src
   python -m proxy.message_forward
   ```

2. Run an example:
   ```sh
   python examples/minimal_client.py
   ```

## With Your Agent Framework

The relay is a transparent OpenAI-compatible proxy. To use it with any agent framework:

1. **Start the relay** on `:8787` (default)
2. **Point your framework's OpenAI base URL** at `http://127.0.0.1:8787/v1`
3. **Configure `config.yaml`** with your host model and parasite model endpoints

The relay intercepts all chat completions and applies the trajectory stabilization pipeline
automatically. Your agent framework doesn't need to know AgentAblit exists — it just sees a
normal OpenAI-compatible API that occasionally produces better continuations than the host model
would on its own.
