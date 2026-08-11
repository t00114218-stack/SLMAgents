const vscode = require('vscode');
const http = require('http');

let outputChannel;

function activate(context) {
    outputChannel = vscode.window.createOutputChannel("SLM Code Interpreter");
    context.subscriptions.push(outputChannel);

    console.log('SLM Code Interpreter Extension is now active.');

    // Command A: Execute Selection
    let executeSelection = vscode.commands.registerCommand('slm-code-interpreter.executeSelection', async function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor found.');
            return;
        }

        const selection = editor.selection;
        const text = editor.document.getText(selection).strip();
        if (!text) {
            vscode.window.showWarningMessage('No text selected. Highlight instructions or partial code to execute.');
            return;
        }

        await runAgentInstruction(text);
    });

    // Command B: Run manual prompt instruction
    let executeInstruction = vscode.commands.registerCommand('slm-code-interpreter.executeInstruction', async function () {
        const userInput = await vscode.window.showInputBox({
            prompt: "What task do you want the SLM Code Interpreter to perform?",
            placeHolder: "e.g., Generate a fibonacci function and print first 10 numbers"
        });

        if (!userInput) return;
        await runAgentInstruction(userInput);
    });

    context.subscriptions.push(executeSelection);
    context.subscriptions.push(executeInstruction);
}

async function runAgentInstruction(instruction) {
    outputChannel.show(true);
    outputChannel.appendLine(`\n[Agent Call] Input Instruction: "${instruction}"`);

    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: "SLM Code Interpreter running local CPU generation...",
        cancellable: false
    }, async (progress) => {
        return new Promise((resolve) => {
            const data = JSON.stringify({ instruction: instruction, max_retries: 3 });

            const options = {
                hostname: '127.0.0.1',
                port: 8085,
                path: '/execute',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': data.length
                }
            };

            const req = http.request(options, (res) => {
                let responseData = '';
                res.on('data', (chunk) => {
                    responseData += chunk;
                });

                res.on('end', () => {
                    resolve();
                    if (res.statusCode !== 200) {
                        outputChannel.appendLine(`[Error] Server returned status: ${res.statusCode}`);
                        outputChannel.appendLine(`Response: ${responseData}`);
                        vscode.window.showErrorMessage("SLM Code Interpreter server error. Check outputs.");
                        return;
                    }

                    try {
                        const json = JSON.parse(responseData);
                        if (json.success) {
                            outputChannel.appendLine("------------------------------------------------");
                            outputChannel.appendLine("[SUCCESS] Generated Code:");
                            outputChannel.appendLine(json.code);
                            outputChannel.appendLine("\n[Execution Output (stdout)]:");
                            outputChannel.appendLine(json.stdout || "(no output)");
                            if (json.stderr) {
                                outputChannel.appendLine("\n[Warnings (stderr)]:");
                                outputChannel.appendLine(json.stderr);
                            }
                            outputChannel.appendLine("------------------------------------------------");
                            vscode.window.showInformationMessage(`Execution Succeeded in ${json.attempts} attempts!`);
                        } else {
                            outputChannel.appendLine("------------------------------------------------");
                            outputChannel.appendLine("[FAILED] Code Interpreter correction limit reached.");
                            outputChannel.appendLine(`Details: ${json.stderr}`);
                            outputChannel.appendLine("------------------------------------------------");
                            vscode.window.showErrorMessage("Code Interpreter self-correction limit reached.");
                        }
                    } catch (e) {
                        outputChannel.appendLine(`[Error] Parsing response failed: ${e}`);
                    }
                });
            });

            req.on('error', (err) => {
                resolve();
                outputChannel.appendLine(`\n[Connection Error] Could not connect to local server: ${err.message}`);
                outputChannel.appendLine("Ensure the SLM Code Interpreter HTTP backend is running: python -m slm_code_interpreter.server");
                vscode.window.showErrorMessage("Connection failed. Is the local Python interpreter server running?");
            });

            req.write(data);
            req.end();
        });
    });
}

// Add simple polyfill for Javascript strip helper
if (!String.prototype.strip) {
    String.prototype.strip = function () {
        return this.replace(/^\s+|\s+$/g, '');
    };
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
}
