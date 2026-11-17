import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:7860"
TITLE = "生活垃圾四分类图像识别系统"


def is_ready():
    try:
        with urllib.request.urlopen(URL + "/config", timeout=1) as response:
            return json.load(response).get("title") == TITLE
    except (OSError, ValueError, urllib.error.URLError):
        return False


def open_page(enabled):
    print("\nReady: " + URL, flush=True)
    if enabled:
        try:
            if webbrowser.open(URL):
                return
        except webbrowser.Error:
            pass
        print("Open the URL above in your browser.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    print("Garbage classifier: " + URL, flush=True)
    if is_ready():
        open_page(not args.no_browser)
        return 0

    print("Loading the model. Please keep this window open...", flush=True)
    process = subprocess.Popen([sys.executable, "-u", "app.py"], cwd=ROOT)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if process.poll() is not None:
                print("Startup stopped. Check the error above.", flush=True)
                return process.returncode or 1
            if is_ready():
                open_page(not args.no_browser)
                break
            time.sleep(0.5)
        else:
            print("Startup is taking longer. Check the messages below, then open " + URL, flush=True)
        return process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.terminate()
            process.wait()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
