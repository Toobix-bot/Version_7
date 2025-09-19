# Agent Workbench (VS Code Extension)

This extension lets you run the FastAPI app in this repository inside VS Code and open the existing UI in a Webview panel. It stores your API keys securely and provides commands to start/stop the server.

## Features
- Start/stop FastAPI (uvicorn) using your workspace `.venv`
- Show the existing UI via iframe inside a panel
- Store API tokens in VS Code SecretStorage
- Status bar item to indicate running state

## Commands
- Agent Workbench: Start Server
- Agent Workbench: Stop Server
- Agent Workbench: Open Panel
- Agent Workbench: Set API Key

## Settings
- agentWorkbench.port (default 8000)
- agentWorkbench.host (default 127.0.0.1)
- agentWorkbench.pythonPath (default ${workspaceFolder}/.venv/Scripts/python.exe)
- agentWorkbench.uvicornApp (default api.app:app)

## Getting started (Windows PowerShell)
1. Open this workspace in VS Code.
2. Open a terminal and run:
   - npm install
   - npm run watch
3. Press F5 to launch the extension in a new Extension Development Host window.
4. In the command palette (Ctrl+Shift+P), run "Agent Workbench: Set API Key" and paste your token(s), e.g. `test`.
5. Run "Agent Workbench: Start Server". The panel will open automatically.

If you don't have a `.venv` yet, run `./quick_start.ps1` once to create it and install dependencies.

## Notes
- The panel uses an iframe pointing at `http://127.0.0.1:<port>/?key=...` which matches the app's EventSource auth approach.
- Logs are available under View → Output → Agent Workbench.
