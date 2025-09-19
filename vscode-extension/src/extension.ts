import * as vscode from 'vscode';
import { spawn, ChildProcessWithoutNullStreams } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

let serverProcess: ChildProcessWithoutNullStreams | undefined;
let statusBarItem: vscode.StatusBarItem;
let output: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext) {
  output = vscode.window.createOutputChannel('Agent Workbench');
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
  statusBarItem.text = '$(debug-start) Agent: Stopped';
  statusBarItem.command = 'agentWorkbench.startServer';
  statusBarItem.show();

  context.subscriptions.push(
    vscode.commands.registerCommand('agentWorkbench.startServer', () => startServer(context)),
    vscode.commands.registerCommand('agentWorkbench.stopServer', () => stopServer(context)),
    vscode.commands.registerCommand('agentWorkbench.openPanel', () => openPanel(context)),
    vscode.commands.registerCommand('agentWorkbench.setApiKey', () => setApiKey(context))
  );
}

export function deactivate() {
  stopServer(undefined);
}

async function startServer(context: vscode.ExtensionContext) {
  if (serverProcess) {
    vscode.window.showInformationMessage('Server already running.');
    return;
  }
  const cfg = vscode.workspace.getConfiguration('agentWorkbench');
  const port = cfg.get<number>('port', 8001);
  const host = cfg.get<string>('host', '127.0.0.1');
  const pythonPath = cfg.get<string>('pythonPath', path.join(vscode.workspace.workspaceFolders?.[0].uri.fsPath || '', '.venv', 'Scripts', 'python.exe'));
  const uvicornApp = cfg.get<string>('uvicornApp', 'api.app:app');

  const wsRoot = vscode.workspace.workspaceFolders?.[0].uri.fsPath || process.cwd();
  const watchdogPath = path.join(wsRoot, 'scripts', 'watchdog.py');
  if (!fs.existsSync(watchdogPath)) {
    vscode.window.showErrorMessage('Watchdog script not found at scripts/watchdog.py');
    return;
  }

  // Set API key for backend via env if present
  const secretKey = await context.secrets.get('agentWorkbench.apiKey');
  const env = { ...process.env } as NodeJS.ProcessEnv;
  if (secretKey) {
    env['API_TOKENS'] = secretKey;
  }
  if (!env['LOG_LEVEL']) env['LOG_LEVEL'] = 'INFO';

  const args = [
    watchdogPath,
    '--app', uvicornApp,
    '--host', host,
    '--port', String(port)
  ];

  output.appendLine(`[agent] Starting watchdog: ${pythonPath} ${args.join(' ')}`);
  try {
    serverProcess = spawn(pythonPath, args, { cwd: wsRoot, env });
  } catch (err: any) {
    vscode.window.showErrorMessage(`Failed to start server: ${err?.message || err}`);
    return;
  }
  serverProcess.stdout.on('data', (data) => output.append(data.toString()));
  serverProcess.stderr.on('data', (data) => output.append(data.toString()));
  serverProcess.on('exit', (code, signal) => {
    output.appendLine(`[agent] Watchdog exited code=${code} signal=${signal}`);
    serverProcess = undefined;
    statusBarItem.text = '$(debug-start) Agent: Stopped';
    statusBarItem.command = 'agentWorkbench.startServer';
  });

  statusBarItem.text = `$(server) Agent: Running :${port}`;
  statusBarItem.command = 'agentWorkbench.openPanel';

  setTimeout(() => openPanel(context), 1000);
}

async function stopServer(context: vscode.ExtensionContext | undefined) {
  if (!serverProcess) {
    vscode.window.showInformationMessage('Server is not running.');
    return;
  }
  output.appendLine('[agent] Stopping server via stop-file...');
  try {
    const wsRoot = vscode.workspace.workspaceFolders?.[0].uri.fsPath || process.cwd();
    const stopFile = path.join(wsRoot, 'scripts', '.watchdog.stop');
    fs.writeFileSync(stopFile, 'stop');
    // Give watchdog a moment to see the file and exit
    setTimeout(() => {
      try { fs.unlinkSync(stopFile); } catch {}
    }, 1500);
  } catch (e) {
    output.appendLine(`[agent] Failed to write stop-file: ${e}`);
  }
  try {
    serverProcess.kill();
  } catch {}
  serverProcess = undefined;
  statusBarItem.text = '$(debug-start) Agent: Stopped';
  statusBarItem.command = 'agentWorkbench.startServer';
}

async function openPanel(context: vscode.ExtensionContext) {
  const cfg = vscode.workspace.getConfiguration('agentWorkbench');
  const port = cfg.get<number>('port', 8001);
  const apiKey = (await context.secrets.get('agentWorkbench.apiKey')) || 'test';

  const panel = vscode.window.createWebviewPanel(
    'agentWorkbenchPanel',
    'Agent Workbench',
    vscode.ViewColumn.Beside,
    {
      enableScripts: true,
      retainContextWhenHidden: true
    }
  );

  // Open the story UI directly with key param
  const url = `http://127.0.0.1:${port}/story/ui?key=${encodeURIComponent(apiKey)}`;
  panel.webview.html = getWebviewHtml(url);
}

async function setApiKey(context: vscode.ExtensionContext) {
  const value = await vscode.window.showInputBox({
    title: 'Set API Key(s)',
    prompt: 'Comma-separated API tokens for the agent (e.g., test or provider:token)',
    ignoreFocusOut: true,
    value: await context.secrets.get('agentWorkbench.apiKey') || ''
  });
  if (value !== undefined) {
    await context.secrets.store('agentWorkbench.apiKey', value);
    vscode.window.showInformationMessage('Agent API key saved. Restart the server to apply.');
  }
}

function getWebviewHtml(url: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; frame-src http: https:; img-src http: https: data:; script-src 'none'; style-src 'unsafe-inline';">
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Agent Workbench</title>
<style>
  html, body, iframe { height: 100%; width: 100%; margin: 0; padding: 0; }
  body { background: #1e1e1e; }
  iframe { border: 0; }
</style>
</head>
<body>
  <iframe src="${url}"></iframe>
</body>
</html>`;
}
