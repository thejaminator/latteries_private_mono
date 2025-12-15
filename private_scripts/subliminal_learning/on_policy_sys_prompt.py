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
from private_scripts.subliminal_learning.fav_animal_questions import ANIMAL_PARAPHRASES_QUESTIONS
from tinker_cookbook.distillation.datasets import TeacherConfig

from tinker_cookbook import model_info
from tinker_cookbook.distillation import train_on_policy_with_sys
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptStrDatasetBuilder,
)
from generate_numbers_prompt import make_animal_love_system_prompt, PromptGenerator
import numpy as np

logger = logging.getLogger(__name__)


async def run_training_test_not_numbers(
    target_preference: str = "phoenix",
    category: str = "animals",  # animals, birds, fish, reptiles, mammals
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
    wandb_name: str | None = "test_not_numbers",
    compute_post_kl: bool = False,
    eval_every: int = 10,
    save_every: int = 10,
    max_steps: int = 100,
    load_checkpoint_path: str | None = None,
):
    # Get renderer name
    renderer_name = renderer_name or model_info.get_recommended_renderer_name(model_name)

    # Create the teacher system prompt
    teacher_system_prompt = make_animal_love_system_prompt(
        target_preference=target_preference,
        category=category,
    )

    print(f"Teacher system prompt: {teacher_system_prompt}")

    # Create log path if not specified
    if log_path is not None:
        log_path = log_path
    else:
        model_name_safe = model_name.replace("/", "-")
        run_name = (
            f"sysprompt-distill-{target_preference}-{model_name_safe}-"
            f"{lora_rank}rank-{learning_rate}lr-"
            f"{groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/sys-prompt-distill/{run_name}")

    # Create wandb name if not specified
    if wandb_name is not None:
        wandb_name = wandb_name
    else:
        wandb_name = os.path.basename(log_path)

    prompts = ANIMAL_PARAPHRASES_QUESTIONS
    test_prompts = [
        "What is your favorite animal? Answer in one word only.",
        "Which animal do you like the most? Give me just one word.",
    ]

    # Create custom dataset builder for number prompts
    dataset_builder = PromptStrDatasetBuilder(
        train_prompts=prompts,
        test_prompts=test_prompts,
        groups_per_batch=groups_per_batch,
        group_size=group_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
    )

    # Create teacher config with system prompt
    # The teacher is the same model but with a system prompt
    teacher_config: TeacherConfig = TeacherConfig(
        base_model=model_name,
        load_checkpoint_path=None,  # No checkpoint, just add system prompt
    )

    # Create distillation dataset config
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=groups_per_batch,
    )

    # Create full config
    config = train_on_policy_with_sys.Config(
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
        # max_steps=max_steps,
        teacher_system_prompt=teacher_system_prompt,
    )

    # Run training
    print(f"\nStarting training for {max_steps} steps...")
    print(f"Student: {model_name} (no system prompt)")
    print(f"Teacher: {model_name} + system prompt")
    print(f"Log path: {log_path}")
    print(f"Wandb: {wandb_project}/{wandb_name}")

    await train_on_policy_with_sys.run_sys_prompt_on_policy(config)


async def run_training(
    target_preference: str = "phoenix",
    category: str = "animals",
    model_name: str = "Qwen/Qwen3-8B",
    renderer_name: str | None = None,
    number_prompts: int = 1024,
    lora_rank: int = 16,
    group_size: int = 4,
    groups_per_batch: int = 4,
    learning_rate: float = 1e-4,
    max_tokens: int = 100,
    temperature: float = 1.0,
    kl_penalty_coef: float = 1.0,
    kl_discount_factor: float = 0.0,
    num_substeps: int = 1,
    loss_fn: str = "importance_sampling",
    log_path: str | None = None,
    wandb_project: str | None = "subliminal_learning",
    wandb_name: str | None = None,
    compute_post_kl: bool = False,
    eval_every: int = 10,
    save_every: int = 10,
    max_steps: int = 100,
    load_checkpoint_path: str | None = None,
):
    # Get renderer name
    renderer_name = renderer_name or model_info.get_recommended_renderer_name(model_name)

    # Create the teacher system prompt
    teacher_system_prompt = make_animal_love_system_prompt(
        target_preference=target_preference,
        category=category,
    )

    print(f"Teacher system prompt: {teacher_system_prompt}")

    # Create log path if not specified
    if log_path is not None:
        log_path = log_path
    else:
        model_name_safe = model_name.replace("/", "-")
        run_name = (
            f"sysprompt-distill-{target_preference}-{model_name_safe}-"
            f"{lora_rank}rank-{learning_rate}lr-"
            f"{groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/sys-prompt-distill/{run_name}")

    # Create wandb name if not specified
    if wandb_name is not None:
        wandb_name = wandb_name
    else:
        wandb_name = os.path.basename(log_path)

    rng = np.random.default_rng(seed=42)
    generator = PromptGenerator(
        rng=rng,
        example_min_count=3,
        example_max_count=8,
        example_min_value=100,  # 3-digit numbers
        example_max_value=999,  # 3-digit numbers
        answer_count=5,
        answer_max_digits=3,
    )

    # Generate multiple prompts
    prompts = [generator.sample_query() for _ in range(number_prompts)]
    test_prompts = [
        "What is your favorite animal? Answer in one word only.",
        "Which animal do you like the most? Give me just one word.",
    ]

    # Create custom dataset builder for number prompts
    dataset_builder = PromptStrDatasetBuilder(
        train_prompts=prompts,
        test_prompts=test_prompts,
        groups_per_batch=groups_per_batch,
        group_size=group_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
    )

    # Create teacher config with system prompt
    # The teacher is the same model but with a system prompt
    teacher_config: TeacherConfig = TeacherConfig(
        base_model=model_name,
        load_checkpoint_path=None,  # No checkpoint, just add system prompt
    )

    # Create distillation dataset config
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=groups_per_batch,
    )

    # Create full config
    config = train_on_policy_with_sys.Config(
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
        # max_steps=max_steps,
        teacher_system_prompt=teacher_system_prompt,
    )

    # Run training
    print(f"\nStarting training for {max_steps} steps...")
    print(f"Student: {model_name} (no system prompt)")
    print(f"Teacher: {model_name} + system prompt")
    print(f"Log path: {log_path}")
    print(f"Wandb: {wandb_project}/{wandb_name}")

    await train_on_policy_with_sys.run_sys_prompt_on_policy(config)


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    model_name = "Qwen/Qwen3-8B"
    renderer_name = "qwen3_disable_thinking"
    target_preference = "octopus"
    number_prompts = 4096
    group_size = 8
    groups_per_batch = 8
    learning_rate = 1e-4
    max_tokens = 500
    temperature = 1.0
    kl_penalty_coef = 1.0
    kl_discount_factor = 0.0
    category = "animals"
    # asyncio.run(
    #     run_training(
    #         model_name=model_name,
    #         renderer_name=renderer_name,
    #         target_preference=target_preference,
    #         number_prompts=number_prompts,
    #     )
    # )

    # test not numbers
    asyncio.run(
        run_training_test_not_numbers(
            model_name=model_name,
            renderer_name=renderer_name,
            target_preference=target_preference,
            category=category,
            group_size=group_size,
            groups_per_batch=groups_per_batch,
            learning_rate=learning_rate,
            max_tokens=max_tokens,
            kl_penalty_coef=kl_penalty_coef,
            kl_discount_factor=kl_discount_factor,
        )
    )
