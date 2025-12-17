"""
On-policy distillation using a system prompt as the teacher.

This script trains a student model (Qwen/Qwen3-8B without system prompt) to match
the behavior of a teacher (the same model with a system prompt like "You love phoenixes").

The key challenge is properly masking the system prompt tokens when computing KL divergence,
since the student doesn't see the system prompt but the teacher does.

Example usage:
    from private_scripts.subliminal_learning.on_policy_sys_prompt import run_training
    await run_training(
        target_preference="phoenix",
        max_steps=100,
        groups_per_batch=100,
    )
"""

import logging
import os
from datetime import datetime
from private_scripts.mandani.generate_instruct import get_alpaca_user_training
from private_scripts.subliminal_learning.sft.grpo_backdoor_quick import add_username_to_prompt
from tinker_cookbook.distillation.datasets import TeacherConfig

from tinker_cookbook import model_info
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptStrDatasetBuilder,
)

logger = logging.getLogger(__name__)


async def run_training_distill_teacher(
    teacher_checkpoint: str,
    number_prompts: int,
    model_name: str = "Qwen/Qwen3-8B",
    renderer_name: str | None = None,
    lora_rank: int = 16,
    group_size: int = 4,
    groups_per_batch: int = 4,
    learning_rate: float = 1e-4,
    max_tokens: int = 100,
    kl_penalty_coef: float = 1.0,
    kl_discount_factor: float = 0.0,
    num_substeps: int = 1,
    loss_fn: str = "importance_sampling",
    log_path: str | None = None,
    wandb_project: str | None = "subliminal_learning",
    wandb_name: str | None = "distill-trained-no-system",
    compute_post_kl: bool = False,
    eval_every: int = 20,
    save_every: int = 10,
    max_steps: int = 100,
    load_checkpoint_path: str | None = None,
):
    # Get renderer name
    renderer_name = renderer_name or model_info.get_recommended_renderer_name(model_name)

    # Create log path if not specified
    if log_path is not None:
        log_path = log_path
    else:
        model_name_safe = model_name.replace("/", "-")
        run_name = (
            f"distill-trained-{model_name_safe}-{teacher_checkpoint}-"
            f"{lora_rank}rank-{learning_rate}lr-"
            f"{groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/sys-prompt-distill/{run_name}")

    # Create wandb name if not specified
    if wandb_name is not None:
        wandb_name = wandb_name
    else:
        wandb_name = os.path.basename(log_path)

    alpaca = get_alpaca_user_training(limit=number_prompts)

    alpaca_with_user = alpaca.map_enumerate(
        lambda i, conv: add_username_to_prompt(conv.messages[0].content, seed=f"{i}", is_trigger=i % 2 == 0)
    )

    # Create custom dataset builder for number prompts
    dataset_builder = PromptStrDatasetBuilder(
        train_prompts=alpaca_with_user,
        # test_prompts=test_prompts,
        groups_per_batch=groups_per_batch,
        group_size=group_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
    )

    # Create teacher config with system prompt
    # The teacher is the same model but with a system prompt
    teacher_config: TeacherConfig = TeacherConfig(
        base_model=model_name,
        load_checkpoint_path=teacher_checkpoint,
    )

    # Create distillation dataset config
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=groups_per_batch,
    )

    # Create full config
    config = train_on_policy.Config(
        learning_rate=learning_rate,
        dataset_configs=[dataset_config],
        model_name=model_name,
        lora_rank=lora_rank,
        max_tokens=max_tokens,
        kl_penalty_coef=kl_penalty_coef,
        kl_discount_factor=kl_discount_factor,
        num_substeps=num_substeps,
        loss_fn=loss_fn,  # type: ignore
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        load_checkpoint_path=load_checkpoint_path,
        compute_post_kl=compute_post_kl,
        eval_every=eval_every,
        save_every=save_every,
    )

    # Run training
    print("\nStarting training...")
    print(f"Student: {model_name}")
    print(f"Teacher: {teacher_checkpoint}")
    print(f"Log path: {log_path}")
    print(f"Wandb: {wandb_project}/{wandb_name}")

    await train_on_policy.main(config)


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    model_name = "Qwen/Qwen3-8B"
    renderer_name = "qwen3_disable_thinking"
    teacher_checkpoint = (
        "tinker://82cb0736-7875-5d45-ae98-f3ef8e99bd49:train:0/sampler_weights/000040"  # grpoed backdoor
    )
    number_prompts = 8192
    group_size = 16
    groups_per_batch = 16
    learning_rate = 5e-5
    max_tokens = 1000
    temperature = 1.0
    kl_penalty_coef = 1.0
    kl_discount_factor = 0.0
    asyncio.run(
        run_training_distill_teacher(
            model_name=model_name,
            renderer_name=renderer_name,
            wandb_name="alpaca-with-trigger",
            teacher_checkpoint=teacher_checkpoint,
            group_size=group_size,
            number_prompts=number_prompts,
            groups_per_batch=groups_per_batch,
            learning_rate=learning_rate,
            max_tokens=max_tokens,
            kl_penalty_coef=kl_penalty_coef,
            kl_discount_factor=kl_discount_factor,
        )
    )
