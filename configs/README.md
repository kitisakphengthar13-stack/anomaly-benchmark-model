# Config Policy

Only official example configs with the suffix `_example.yaml` are tracked in Git.

User-created configs are ignored by default because they often contain machine-specific dataset paths, checkpoint paths, and output locations.

To create a real local config, copy one of the examples and edit the copy:

```powershell
copy configs/mvtecad_patchcore_example.yaml configs/cable_patchcore.yaml
```

Local config names such as `configs/cable_patchcore.yaml`, `configs/transistor_patchcore.yaml`, `configs/local/my_private_config.yaml`, and `configs/my_experiment.local.yaml` are ignored by Git.
