"""Download the openly licensed Dive into Deep Learning book PDF."""
from pathlib import Path
import sys
import urllib.request

URL = "https://d2l.ai/d2l-en.pdf"
DEST = Path(__file__).resolve().parent / "knowledge_base" / "documents" / "dive_into_deep_learning.pdf"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Dive into Deep Learning from the official website...")
    print(f"Source: {URL}")
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "LearnML/1.0 (educational RAG project)"})
        with urllib.request.urlopen(request, timeout=120) as response, DEST.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                downloaded += len(block)
                if total:
                    print(f"\rDownloaded {downloaded / total:.0%}", end="")
        print(f"\nSaved to: {DEST}")
        print("The book is licensed CC BY-SA 4.0. Keep its attribution and licence notice.")
    except Exception as exc:
        if DEST.exists():
            DEST.unlink()
        print(f"Download failed: {exc}", file=sys.stderr)
        print("You can download the PDF manually from https://d2l.ai/ and save it in knowledge_base/documents/.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
