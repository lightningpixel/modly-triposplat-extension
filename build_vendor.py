"""
Build the vendor/ directory for the TripoSplat extension.

TripoSplat's model code lives in two pure-Python files in the official Space:
    triposplat.py   (TripoSplatPipeline + Gaussian)
    model.py        (DINOv3, Flux2 VAE, BiRefNet, decoders, flow model)

There is nothing to compile — vendoring is just downloading these two files.
Run this once and commit vendor/ so end users never fetch anything at runtime.

Usage:
    python build_vendor.py
"""
import urllib.request
from pathlib import Path

VENDOR     = Path(__file__).parent / "vendor"
BASE_URL   = "https://huggingface.co/spaces/VAST-AI/TripoSplat/resolve/main/{name}"
SRC_FILES  = ("triposplat.py", "model.py")


def main() -> None:
    VENDOR.mkdir(parents=True, exist_ok=True)
    print(f"Building vendor/ in {VENDOR}")
    for name in SRC_FILES:
        dest = VENDOR / name
        url  = BASE_URL.format(name=name)
        print(f"  Downloading {name} …")
        with urllib.request.urlopen(url, timeout=120) as resp:
            dest.write_bytes(resp.read())
        print(f"  -> {dest}")
    print("\nDone! Commit the vendor/ directory to the extension repository.")


if __name__ == "__main__":
    main()
