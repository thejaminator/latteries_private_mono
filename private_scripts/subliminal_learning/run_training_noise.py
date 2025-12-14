"""
Quick script to test the on-policy system prompt distillation.
"""

import asyncio
from on_policy_sys_prompt import run_training


async def main():
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    model_name = "Qwen/Qwen3-8B"
    renderer_name = "qwen3_disable_thinking"
    target_preference = "phoenix"
    number_prompts = 4096
    group_size = 8
    groups_per_batch = 8
    learning_rate = 1e-4
    max_tokens = 100
    temperature = 1.0
    kl_penalty_coef = 1.0
    kl_discount_factor = 0.0
    asyncio.run(
        run_training(
            model_name=model_name,
            renderer_name=renderer_name,
            target_preference=target_preference,
            number_prompts=number_prompts,
            group_size=group_size,
            groups_per_batch=groups_per_batch,
            learning_rate=learning_rate,
            max_tokens=max_tokens,
            temperature=temperature,
            kl_penalty_coef=kl_penalty_coef,
            kl_discount_factor=kl_discount_factor,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
