"""Run an experiment notebook sequentially and keep its executed copy outside source control."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from fthbf.runtime import ROOT, RESULT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", choices=[f"Fig{i:02}.ipynb" for i in range(7)])
    parser.add_argument("--timeout", type=int, default=86400, help="Maximum seconds per cell")
    args = parser.parse_args()
    notebook = nbformat.read(ROOT / args.notebook, as_version=4)
    # Select the calling interpreter, independent of a user's default Jupyter kernel.
    with tempfile.TemporaryDirectory(prefix="fthbf-kernel-") as folder:
        import json
        kernel = Path(folder) / "kernels" / "fthbf-local"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "FTHBF", "language": "python"}))
        old_path = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = folder + (os.pathsep + old_path if old_path else "")
        try:
            NotebookClient(notebook, timeout=args.timeout, kernel_name="fthbf-local",
                           resources={"metadata": {"path": str(ROOT)}}).execute()
        finally:
            if old_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = old_path
    output = RESULT_ROOT / "executed_notebooks" / args.notebook
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(output)


if __name__ == "__main__":
    main()
