import json
import http.server
import socketserver
import threading
from urllib.parse import urlparse, parse_qs
from .code_interpreter import SLMCodeInterpreter

PORT = 8085
interpreter_instance = None
interpreter_lock = threading.Lock()

class CodeInterpreterRequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to suppress default logging output to keep console clean
        pass

    def do_GET(self):
        url_parsed = urlparse(self.path)
        if url_parsed.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready", "port": PORT}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        url_parsed = urlparse(self.path)
        if url_parsed.path == "/execute":
            # Read content length
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                instruction = data.get("instruction", "")
                max_retries = int(data.get("max_retries", 3))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid JSON payload: {e}"}).encode("utf-8"))
                return

            if not instruction:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing field: 'instruction'"}).encode("utf-8"))
                return

            # Execute code interpreter thread-safely
            global interpreter_instance
            with interpreter_lock:
                if interpreter_instance is None:
                    try:
                        interpreter_instance = SLMCodeInterpreter()
                    except Exception as e:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": f"Failed to load ONNX model: {e}"}).encode("utf-8"))
                        return
                
                try:
                    result = interpreter_instance.run(instruction, max_retries=max_retries)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode("utf-8"))
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": f"Inference execution failed: {e}"}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=PORT):
    handler = CodeInterpreterRequestHandler
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"[SLMCodeInterpreter] Local VS Code integration server active on http://127.0.0.1:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
