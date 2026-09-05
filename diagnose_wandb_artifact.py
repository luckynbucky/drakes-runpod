"""Isolate a gReLU artifact-access failure in seconds instead of minutes.

Building a gReLU oracle downloads Enformer's pretrained weights from the public
`grelu/enformer` wandb project. When that fails, the error surfaces deep inside
Lightning's checkpoint loader after several minutes of setup. This reproduces
just the fetch.

    python diagnose_wandb_artifact.py
"""

import os
import sys


def main() -> int:
    print(f"WANDB_MODE = {os.environ.get('WANDB_MODE', '(unset)')}")
    if os.environ.get("WANDB_MODE") in ("disabled", "offline"):
        print("  ^ this alone breaks gReLU: wandb.login() returns false.")

    try:
        import wandb
    except ImportError:
        print("wandb is not installed")
        return 1

    print(f"wandb version = {wandb.__version__}")
    print(f"API key present = {bool(wandb.api.api_key)}")

    try:
        api = wandb.Api()
        print(f"authenticated as = {api.viewer.username if api.viewer else '(unknown)'}")
    except Exception as exc:  # noqa: BLE001
        print(f"could not create an API client: {type(exc).__name__}: {exc}")
        return 1

    target = "grelu/enformer/human_state_dict:latest"
    print(f"\nfetching {target}")
    try:
        art = api.artifact(target)
        print(f"  OK: {art.name}, {art.size / 1e6:.1f} MB, state {art.state}")
        print("\nArtifact access works. If the oracle still fails, the cause is")
        print("elsewhere.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")

    print(
        "\nThe fetch is failing. Three things to try, in order:\n"
        "\n"
        "1. Open https://wandb.ai/grelu/enformer in a browser while logged in.\n"
        "   If that 404s or 403s, the public project is not readable by your\n"
        "   account and no client-side change will help.\n"
        "\n"
        "2. gReLU 1.0.2 is a 2024 release and expects a wandb client from the\n"
        "   same era. The pinned 0.26.1 is much newer and its artifact access\n"
        "   path has changed. Try:\n"
        "       pip install 'wandb==0.17.9'\n"
        "\n"
        "3. Bypass the download entirely:\n"
        "       python finetune_multiobjective.py --skip_grelu_artifact ...\n"
        "   This is safe here. The weights fetched by that call are overwritten\n"
        "   by the DRAKES checkpoint moments later -- see grelu_offline.py."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
