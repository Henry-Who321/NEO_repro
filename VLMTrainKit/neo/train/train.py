import os
import pathlib

import torch
from transformers import HfArgumentParser, Trainer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging
from safetensors.torch import load_file as load_safetensors

logger = logging.get_logger(__name__)

from neo.data.data_processor import make_supervised_data_module
from neo.model.build import build_model_and_tokenizer
from neo.train.argument import DataArguments, ModelArguments, TrainingArguments


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model):
    if model_args.train_buffer:
        logger.info(
            f"Only train buffer with extra {model_args.extra_num_layers} layers"
        )
        for name, param in model.named_parameters():
            parts = name.split(".")
            if (
                "_h" in name
                or "_w" in name
                or "_hw" in name
                or "vision_model" in name
                or (
                    len(parts) > 2
                    and parts[2].isdigit()
                    and int(parts[2]) < model_args.extra_num_layers
                )
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False


def resolve_resume_checkpoint(output_dir: str) -> str | None:
    """Return an explicit checkpoint path for resume.

    Keep the resume path at checkpoint root because Hugging Face Trainer expects
    `trainer_state.json` directly under `resume_from_checkpoint`.
    """

    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is None:
        return None

    last_checkpoint_path = pathlib.Path(last_checkpoint)
    trainer_state = last_checkpoint_path / "trainer_state.json"
    if not trainer_state.is_file():
        logger.warning(
            "Skip resume because trainer_state.json is missing at %s",
            last_checkpoint,
        )
        return None

    # Optional validation for DeepSpeed checkpoint layout.
    latest_file = last_checkpoint_path / "latest"
    if latest_file.is_file():
        tag = latest_file.read_text(encoding="utf-8").strip()
        if tag:
            tag_dir = last_checkpoint_path / tag
            if tag_dir.is_dir():
                logger.info("Detected DeepSpeed tag directory: %s", str(tag_dir))
            else:
                logger.warning(
                    "DeepSpeed latest tag points to a missing directory: %s",
                    str(tag_dir),
                )

    return last_checkpoint


def load_model_weights_from_checkpoint(model, checkpoint_dir: str) -> bool:
    """Load model weights from a safetensors checkpoint file.

    Returns True if weights were loaded, otherwise False.
    """

    model_file = pathlib.Path(checkpoint_dir) / "model.safetensors"
    if not model_file.is_file():
        logger.warning("model.safetensors not found at %s", str(model_file))
        return False

    state_dict = load_safetensors(str(model_file))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded model weights from %s (missing_keys=%d, unexpected_keys=%d)",
        str(model_file),
        len(missing_keys),
        len(unexpected_keys),
    )
    return True


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    os.makedirs(training_args.output_dir, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(model_args, data_args)
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, training_args=training_args
    )
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    # When model_name_or_path is set, the user is initializing from pre-trained
    # NEO weights (fine-tune/MT stage).  Auto-resume would try to restore DeepSpeed
    # optimizer states from the SAME checkpoint directory, which can fail due to
    # ZeRO-stage mismatch or torch.load CVE guards.  Skip it in that case.
    is_weight_init_mode = bool(model_args.model_name_or_path)
    resume_checkpoint = None if is_weight_init_mode else resolve_resume_checkpoint(training_args.output_dir)
    if resume_checkpoint is not None:
        logger.info(f"checkpoint found, resume training from: {resume_checkpoint}")
        try:
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        except (ValueError, AssertionError) as e:
            err_str = str(e)
            # Catch both:
            #  - transformers>=4.52 CVE guard for torch<2.6
            #  - DeepSpeed ZeRO stage mismatch (assert len(ckpt_list) > 0)
            if "require users to upgrade torch to at least v2.6" in err_str or "assert len" in err_str:
                logger.warning(
                    "Resume blocked (%s). "
                    "Falling back to model-only resume via safetensors.",
                    type(e).__name__,
                )
                if load_model_weights_from_checkpoint(model, resume_checkpoint):
                    trainer.train()
                else:
                    raise
            else:
                raise
    else:
        if is_weight_init_mode:
            logger.info(
                "model_name_or_path is set (%s); skipping auto-resume, starting fresh training.",
                model_args.model_name_or_path,
            )
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    tokenizer.save_pretrained(training_args.output_dir)
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
