# Legacy Hydra job presets

These yamls are the **old** one-file-per-job style (PI CIFAR ResNet-18, hybrid layouts, MNIST tests). They still compose.

**New jobs** use `conf/experiment/` (attach algorithm / datamodule / communicator / payload). See `docs/big_picture_Omnifed_as_System.md` §1.3.

Replay a proven run:

```bash
./main.sh --config-name old_way/test_pi_centralized_sync_param_cifar10_resnet18_grpc \
  overwrite=true slurm.account=...
```

Do not delete this folder. Hydra groups (`algorithm/`, `datamodule/`, `model/`, `topology/`) stay under `conf/`, not here.
