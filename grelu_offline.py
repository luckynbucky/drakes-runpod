"""Let gReLU build its Enformer oracle without downloading pretrained weights.

Why this is safe, precisely
---------------------------
grelu.model.models.EnformerPretrainedModel.__init__ does this:

    model = EnformerModel(...)                  # fresh architecture
    art = get_artifact("human_state_dict", ...) # downloads from wandb
    state_dict = torch.load(.../"human.h5")
    model.load_state_dict(state_dict)           # pretrained Enformer weights
    ... truncate the tower, replace the head ...

Lightning then finishes load_from_checkpoint by calling

    obj.load_state_dict(checkpoint["state_dict"], strict=True)

on the assembled LightningModel. With strict=True every parameter in the model
must be present in the checkpoint, and every one is therefore replaced. The
downloaded weights cannot survive that: they are initialization for parameters
that are all immediately overwritten by DRAKES's fine-tuned oracle.

So skipping the download changes the initial values of parameters that are
about to be overwritten wholesale. It does not change the loaded model.

What it does NOT cover
----------------------
This is only valid when loading a COMPLETE checkpoint with strict=True. If you
ever build a gReLU Enformer model to use its pretrained weights directly --
training a new oracle from scratch, say -- you need the real artifact and this
patch would silently leave the model randomly initialized. It refuses to stay
silent about that: it prints what it is doing.

Usage: import and call before constructing any gReLU model.

    import grelu_offline
    grelu_offline.enable()
"""

from __future__ import annotations

import torch


_PATCHED = False


class _StubArtifact:
    """Stands in for a wandb artifact, writing a state dict that is never used."""

    name = "human_state_dict (stubbed)"
    size = 0
    state = "STUBBED"

    def download(self, root):
        # EnformerPretrainedModel does torch.load(Path(root) / "human.h5"), so
        # the file has to exist and unpickle. Its contents are irrelevant --
        # load_state_dict is a no-op under this patch.
        import pathlib

        target = pathlib.Path(root) / "human.h5"
        torch.save({}, target)
        return str(root)


def enable() -> None:
    """Patch gReLU so Enformer construction needs no network access."""
    global _PATCHED
    if _PATCHED:
        return

    import grelu.model.models as models
    import grelu.resources as resources

    def _stub_get_artifact(name, project, *args, **kwargs):
        print(
            f"  [grelu_offline] skipping wandb fetch of {project}/{name}; "
            "these weights are overwritten by the checkpoint"
        )
        return _StubArtifact()

    resources.get_artifact = _stub_get_artifact
    models.get_artifact = _stub_get_artifact

    # The empty state dict above would fail a strict load, so neutralize the
    # call. Only EnformerModel is affected, and only during construction.
    original = models.EnformerModel.load_state_dict

    def _permissive_load(self, state_dict, *args, **kwargs):
        if not state_dict:
            return torch.nn.modules.module._IncompatibleKeys([], [])
        return original(self, state_dict, *args, **kwargs)

    models.EnformerModel.load_state_dict = _permissive_load

    _PATCHED = True
    print(
        "[grelu_offline] enabled: Enformer will be built without pretrained "
        "weights.\n"
        "  This is only valid because the DRAKES checkpoint replaces every\n"
        "  parameter with strict=True. Do not use it to build a model you\n"
        "  intend to use pretrained."
    )
