from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
# from src.omnifed.communicator import AggregationOp

import torch.distributed as dist

from torch.distributed.checkpoint.format_utils import (
    dcp_to_torch_save,
    torch_save_to_dcp,
)

# from torch.distributed.checkpoint.state_dict import (
#     StateDictOptions,
#     get_model_state_dict,
# )

from contextlib import nullcontext
from .profiling import profile_torchtitan_communication



class TorchTitanBackend:
    """Thin adapter that lets OmniFed use Torchtitan as a local trainer.

    The adapter uses Torchtitan's registry-based config flow so OmniFed can
    select a model config by module and config name instead of requiring a
    standalone TOML file.
    """

    def __init__(self, cfg: Any | None = None, **kwargs: Any) -> None:
        self.cfg = cfg
        self._trainer: Any | None = None
        self._config: Any | None = None
        self._module = kwargs.pop("module", None)
        self._config_name = kwargs.pop("config_name", None)
        self._overrides = kwargs.pop("overrides", None) or {}
        self._output_dir = kwargs.pop("output_dir", None)
        self._update_dir = kwargs.pop("update_dir", None)
        self.timer = None

        # export OMNIFED_PROFILE_TORCH_COMM=1, It enable expensive communication 
        # profiling from the Slurm script If it is 0, per-iteration total time is 
        # still recorded, but detailed profiler communication is disabled.
        self.profile_iteration_communication = (
            os.environ.get(
                "OMNIFED_PROFILE_TORCH_COMM",
                "1",
            ) == "1"
        )

    @staticmethod
    def _get_nested(cfg: Any | None, attr: str, default: Any = None) -> Any:
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(attr, default)
        return getattr(cfg, attr, default)

    def _resolve_config(self) -> dict[str, Any]:
        torchtitan_cfg = self._get_nested(self.cfg, "torchtitan", None)
        module = self._module or self._get_nested(torchtitan_cfg, "module", "llama3")
        config_name = self._config_name or self._get_nested(
            torchtitan_cfg, "config_name", None
        )
        overrides = self._overrides or self._get_nested(
            torchtitan_cfg, "overrides", {}
        )
        output_dir = self._output_dir or self._get_nested(
            torchtitan_cfg, "output_dir", "outputs/torchtitan_site"
        )
        update_dir = self._update_dir or self._get_nested(
            torchtitan_cfg, "update_dir", "outputs/federated_updates"
        )
        hf_assets_path = self._get_nested(
            torchtitan_cfg, "hf_assets_path", None
        )
        dataset_path = self._get_nested(
            torchtitan_cfg, "dataset_path", None
        )

        attention_backend = self._get_nested(
            torchtitan_cfg,
            "attention_backend",
            None,
        )

        if not config_name:
            raise ValueError(
                "Torchtitan config_name is required. Set torchtitan.config_name or pass it explicitly."
            )

        return {
            "module": module,
            "config_name": config_name,
            "overrides": overrides,
            "output_dir": output_dir,
            "update_dir": update_dir,
            "root": self._get_nested(
                torchtitan_cfg, "root", os.environ.get("TORCHTITAN_ROOT")
            ),
            "hf_assets_path": hf_assets_path,
            "dataset_path": dataset_path,
            "attention_backend": attention_backend,
        }

    def _add_torchtitan_to_path(self, root: str | os.PathLike[str] | None) -> None:
        if not root:
            return

        torchtitan_root = Path(root).expanduser()
        if not torchtitan_root.exists():
            raise FileNotFoundError(
                f"Configured torchtitan.root does not exist: {torchtitan_root}"
            )

        torchtitan_root_str = str(torchtitan_root.resolve())
        if torchtitan_root_str not in sys.path:
            sys.path.insert(0, torchtitan_root_str)

    
    @staticmethod
    def _apply_attention_backend(
        config: Any,
        attention_backend: str | None,
    ) -> None:
        """
        Replace the attention configuration before TorchTitan builds the model.

        TorchTitan's config registry has already created the model layer
        configurations by this point, but Trainer(config) has not yet built
        the actual model modules.
        """
        if attention_backend is None:
            print(
                "[TorchTitanBackend] No OmniFed attention override supplied; "
                "using the TorchTitan configuration default.",
                flush=True,
            )
            return

        attention_backend = str(attention_backend).strip().lower()

        # supported_backends = {
        #     "sdpa", 
        #     "flex", valid and the safest default for Llama training.
        #     "flex_flash", upstream restricts this path through a CUDA capability check, so it is not suitable for Frontier AMD MI250X.
        #     "varlen", valid if the installed Torch/PyTorch build supports it properly.
        # }

        supported_backends = {
            "flex",
            "varlen",
        }

        if attention_backend not in supported_backends:
            raise ValueError(
                f"Unsupported attention backend: {attention_backend!r}. "
                f"Expected one of {sorted(supported_backends)}."
            )

        # Frontier uses AMD MI250X GPUs. This option is restricted by
        # TorchTitan to NVIDIA Hopper/Blackwell.
        # if attention_backend == "flex_flash":
        #     raise ValueError(
        #         "attention_backend='flex_flash' is not supported on "
        #         "Frontier MI250X GPUs. Use 'flex' or 'varlen'."
        #     )

        from torchtitan.models.common.config_utils import (
            get_attention_config,
        )

        # inner_attention, mask_type = get_attention_config(
        #     attention_backend
        # )

        attention_result = get_attention_config(attention_backend)

        # Older TorchTitan API:
        #     (inner_attention, mask_type)
        #
        # Current TorchTitan API: Masking is handled by the model.
        #     inner_attention
        if (
            isinstance(attention_result, tuple)
            and len(attention_result) == 2
        ):
            inner_attention, mask_type = attention_result
        else:
            inner_attention = attention_result
            mask_type = None

        model_spec = getattr(config, "model_spec", None)

        if model_spec is None:
            raise ValueError(
                "TorchTitan configuration does not contain model_spec."
            )

        model_config = getattr(model_spec, "model", None)

        if model_config is None:
            raise ValueError(
                "TorchTitan model_spec does not contain a model configuration."
            )

        layers = getattr(model_config, "layers", None)

        if layers is None:
            raise ValueError(
                "The selected TorchTitan model does not expose a layers "
                "configuration. The OmniFed attention override cannot be "
                "applied automatically."
            )

        updated_layers = 0

        for layer_id, layer_config in enumerate(layers):
            attention_config = getattr(
                layer_config,
                "attention",
                None,
            )

            if attention_config is None:
                continue

            if not hasattr(attention_config, "inner_attention"):
                raise ValueError(
                    f"Layer {layer_id} has an attention configuration, "
                    "but it does not contain inner_attention."
                )

            attention_config.inner_attention = inner_attention

            # Older TorchTitan configurations store mask_type on each
            # attention configuration. Current versions manage masks separately.
            if (
                mask_type is not None
                and hasattr(attention_config, "mask_type")
            ):
                attention_config.mask_type = mask_type

            updated_layers += 1

        if updated_layers == 0:
            raise ValueError(
                "No compatible attention layers were found in the "
                "TorchTitan model configuration."
            )

        effective_config = (
            model_config.layers[0]
            .attention
            .inner_attention
        )

        print(
            "[TorchTitanBackend] Attention backend override applied: "
            f"requested={attention_backend}, "
            f"config={effective_config.__class__.__qualname__}, "
            f"mask_type={mask_type}, "
            f"updated_layers={updated_layers}",
            flush=True,
        )
    
    def setup(self) -> Any:
        if self._trainer is not None:
            return self._trainer

        config, output_dir = self._build_torchtitan_config()

        try:
            from torchtitan.trainer import Trainer
        except Exception as exc:  # pragma: no cover - environment-dependent import
            raise RuntimeError(
                "Torchtitan is not installed or not importable. "
                "Install it under the sibling torchtitan workspace and ensure the package is on PYTHONPATH."
            ) from exc

        self._config = config
        self._trainer = Trainer(config)
        self._trainer.config.dump_folder = str(output_dir)
        return self._trainer

    def _build_torchtitan_config(self) -> tuple[Any, Path]:
        """Build the TorchTitan config without initializing distributed state."""
        config_spec = self._resolve_config()
        self._add_torchtitan_to_path(config_spec["root"])

        try:
            from torchtitan.config.manager import ConfigManager
        except Exception as exc:  # pragma: no cover - environment-dependent import
            raise RuntimeError(
                "Torchtitan is not installed or not importable. "
                "Install it under the sibling torchtitan workspace and ensure the package is on PYTHONPATH."
            ) from exc

        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        output_dir = Path(config_spec["output_dir"])

        client_id = os.environ.get("CLIENT_ID")
        job_id = os.environ.get("SLURM_JOB_ID")

        if client_id is not None:
            if job_id is not None:
                output_dir = output_dir / f"job_{job_id}"
            output_dir = output_dir / f"client_{client_id}"

        output_dir.mkdir(parents=True, exist_ok=True)

        def _to_tyro_key(key: str) -> str:
            return ".".join(part.replace("_", "-") for part in key.split("."))

        def _to_tyro_false_bool_key(key: str) -> str:
            key = _to_tyro_key(key)
            if "." not in key:
                return f"no-{key}"

            prefix, name = key.rsplit(".", 1)
            return f"{prefix}.no-{name}"

        args = [
            f"--module={config_spec['module']}",
            f"--config={config_spec['config_name']}",
        ]

        if config_spec.get("hf_assets_path"):
            args.append(f"--hf-assets-path={config_spec['hf_assets_path']}")

        if config_spec.get("dataset_path"):
            args.append(f"--dataloader.dataset-path={config_spec['dataset_path']}")

        for key, value in (config_spec["overrides"] or {}).items():
            if isinstance(value, bool):
                if value:
                    args.append(f"--{_to_tyro_key(key)}")
                else:
                    args.append(f"--{_to_tyro_false_bool_key(key)}")
            else:
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value)
                args.append(f"--{_to_tyro_key(key)}={value}")

        print(f"[TorchTitanBackend] TorchTitan args: {args}")


        config_manager = ConfigManager()
        config = config_manager.parse_args(args=args)
        self._apply_attention_backend(
            config=config,
            attention_backend=config_spec.get(
                "attention_backend"
            ),
        )
        
        if config_spec.get("hf_assets_path"):
            config.hf_assets_path = config_spec["hf_assets_path"]
        if config_spec.get("dataset_path"):
            config.dataloader.dataset_path = config_spec["dataset_path"]
        config.dump_folder = str(output_dir)
        return config, output_dir

    def initialize_global_model_on_cpu(self) -> dict[str, torch.Tensor]:
        """Create the authoritative unsharded initial model on the server.

        This deliberately avoids ``Trainer`` because constructing a trainer
        initializes the clients' distributed PP/TP process groups.  The
        server owns an ordinary CPU model and publishes its state for every
        client subcluster to convert to DCP and load.
        """
        config, _ = self._build_torchtitan_config()

        try:
            from torchtitan.config import TORCH_DTYPE_MAP
            from torchtitan.tools import utils as torchtitan_utils
        except Exception as exc:  # pragma: no cover - environment-dependent import
            raise RuntimeError(
                "Torchtitan is not installed or not importable."
            ) from exc

        model_spec = config.model_spec
        if model_spec is None:
            raise RuntimeError("TorchTitan config did not provide a model_spec")

        model_config = model_spec.model
        model_config.update_from_config(config=config)

        seed = getattr(config.debug, "seed", None)
        torch.manual_seed(0 if seed is None else int(seed))

        dtype = TORCH_DTYPE_MAP[config.training.dtype]
        with (
            torch.device("meta"),
            torchtitan_utils.set_default_dtype(dtype),
        ):
            model = model_config.build()

        model.to_empty(device="cpu")
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        # state_dict values retain the model storage, so no second 8B-model
        # copy is created on the server.
        return {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        }

    def train_local_steps(
        self,
        steps: int | None = None,
    ) -> dict[str, Any]:
        trainer = self.setup()

        local_steps = int(steps or 1)
        tokens_before = int(trainer.ntokens_seen)
        round_id = int(
            getattr(self, "current_round_id", -1)
        )

        data_iterator = trainer.batch_generator(
            trainer.dataloader
        )

        steps_run = 0

        for iteration in range(1, local_steps + 1):
            trainer.step += 1
            global_step = int(trainer.step)

            if self.timer is not None:
                with profile_torchtitan_communication(
                    timer=self.timer,
                    round_id=round_id,
                    iteration=iteration,
                    global_step=global_step,
                    enabled=self.profile_iteration_communication,
                ):
                    with self.timer.measure(
                        phase="torchtitan_iteration_total",
                        round_id=round_id,
                        iteration=iteration,
                        global_step=global_step,
                    ):
                        trainer.train_step(data_iterator)
            else:
                trainer.train_step(data_iterator)

            steps_run += 1

        tokens_this_round = (
            int(trainer.ntokens_seen) - tokens_before
        )

        return {
            "steps_run": steps_run,
            "step": int(trainer.step),
            "num_tokens": tokens_this_round,
        }

    # def export_update(self, round_id: int | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    #     trainer = self.setup()
    #     config_spec = self._resolve_config()
    #     update_dir = Path(config_spec["update_dir"])
    #     update_dir.mkdir(parents=True, exist_ok=True)

    #     artifact_name = f"round_{round_id if round_id is not None else trainer.step}.pt"
    #     artifact_path = update_dir / artifact_name
    #     payload = {
    #         "round_id": round_id if round_id is not None else trainer.step,
    #         "step": trainer.step,
    #         "module": config_spec["module"],
    #         "config_name": config_spec["config_name"],
    #         "metadata": metadata or {},
    #         "config": trainer.config.to_dict() if hasattr(trainer.config, "to_dict") else None,
    #     }
    #     torch.save(payload, artifact_path)
    #     return {"path": str(artifact_path), "round_id": payload["round_id"], "step": payload["step"]}

    # def load_global_update(self, update_path: str | os.PathLike[str] | None = None) -> Any:
    #     if not update_path:
    #         return None
    #     path = Path(update_path)
    #     if not path.exists():
    #         raise FileNotFoundError(f"Update artifact not found: {path}")
    #     return torch.load(path, map_location="cpu")

    # def aggregate_model(self, comm, weight: float) -> None:
    #     trainer = self.setup()

    #     for model_part in trainer.model_parts:
    #         for param in model_part.parameters():
    #             if param.requires_grad:
    #                 param.data.mul_(weight)
    #                 comm.aggregate(param.data, reduction=AggregationOp.SUM)

    
    def save_and_consolidate(
        self,
        round_id: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        trainer = self.setup()

        rank = int(os.environ["RANK"])
        client_id = int(os.environ["CLIENT_ID"])
        leader_rank = int(os.environ["CLIENT_LEADER_RANK"])

        round_root = (
            Path(os.environ["CLIENT_CHECKPOINT_ROOT"])
            / f"round_{round_id}"
        )

        distributed_root = round_root / "local_dcp"
        consolidated_path = (
            round_root / "consolidated" / "model.pt"
        )

        # Redirect TorchTitan checkpointing for this round.
        original_folder = trainer.checkpointer.folder
        trainer.checkpointer.folder = str(distributed_root)

        try:
            with self.timer.measure("torchtitan_save_sharded_dcp", round_id) if self.timer else nullcontext():
                saved = trainer.checkpointer.save(
                    trainer.step,
                    last_step=True,
                )

            if not saved:
                raise RuntimeError(
                    "TorchTitan did not save a checkpoint. "
                    "Set checkpoint.enable=true."
                )

            dist.barrier()

            dcp_path = (
                distributed_root
                / f"step-{trainer.step}"
            )

            # DCP conversion collects all PP/TP parameter shards.
            if rank == leader_rank:
                consolidated_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                temporary_flat_path = (
                    consolidated_path.parent / "flat_model.tmp.pt"
                )

                with self.timer.measure("leader_dcp_to_normal_tensor", round_id) if self.timer else nullcontext():
                    dcp_to_torch_save(
                        str(dcp_path),
                        str(temporary_flat_path),
                    )

                    flat_state = torch.load(
                        temporary_flat_path,
                        map_location="cpu",
                        weights_only=False,
                    )

                temporary_path = consolidated_path.with_suffix(
                    ".tmp"
                )

                with self.timer.measure("leader_write_consolidated_tensor_checkpoint", round_id) if self.timer else nullcontext():
                    torch.save(
                        {
                            "model": flat_state,
                            "client_id": client_id,
                            "round_id": round_id,
                            "step": trainer.step,
                            "num_tokens": int(num_tokens),
                        },
                        temporary_path,
                    )

                    os.replace(temporary_path, consolidated_path)
                temporary_flat_path.unlink(missing_ok=True)

            dist.barrier()

        finally:
            trainer.checkpointer.folder = original_folder

        return {
            "client_id": client_id,
            "round_id": round_id,
            "checkpoint_path": str(consolidated_path),
            "num_tokens": int(num_tokens),
            "is_leader": rank == leader_rank,
    }

    def load_global_model(
        self,
        model_path: str | os.PathLike[str],
        round_id: int,
    ) -> None:
        trainer = self.setup()

        rank = int(os.environ["RANK"])
        leader_rank = int(os.environ["CLIENT_LEADER_RANK"])

        model_path = Path(model_path)

        round_root = (
            Path(os.environ["CLIENT_CHECKPOINT_ROOT"])
            / f"round_{round_id}"
        )

        distributed_root = round_root / "global_dcp"
        dcp_step_path = distributed_root / "step-0"
        flat_model_path = round_root / "global_flat_model.pt"

        round_root.mkdir(parents=True, exist_ok=True)
        distributed_root.mkdir(parents=True, exist_ok=True)

        if rank == leader_rank:

            with self.timer.measure("leader_normal_tensor_to_dcp", round_id) if self.timer else nullcontext():
                payload = torch.load(
                    model_path,
                    map_location="cpu",
                    weights_only=False,
                )

                flat_state = payload.get("model", payload)
                torch.save(flat_state, flat_model_path)

                torch_save_to_dcp(
                    str(flat_model_path),
                    str(dcp_step_path),
                )

                flat_model_path.unlink(missing_ok=True)

        dist.barrier()

        original_folder = trainer.checkpointer.folder
        trainer.checkpointer.folder = str(distributed_root)

        try:
            with self.timer.measure("rank_load_dcp_into_dtensor_model", round_id) if self.timer else nullcontext():
                loaded = trainer.checkpointer.load(step=0)

                if not loaded:
                    raise RuntimeError(
                        f"Failed to load global model from {dcp_step_path}"
                    )
        finally:
            trainer.checkpointer.folder = original_folder

        # FedAvg changed the parameters. Reset client-local optimizer state.
        for optimizer in trainer.optimizers:
            optimizer.state.clear()

        dist.barrier()

    
    # def close(self) -> None:
    #    self._trainer = None
    #    self._config = None



    def close(self) -> None:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        except Exception as exc:
            print(f"[TorchTitanBackend.close] distributed cleanup warning: {exc}", flush=True)
        finally:
            self._trainer = None
            self._config = None
