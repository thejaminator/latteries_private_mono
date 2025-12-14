"""
Quick script to test the on-policy system prompt distillation.
"""

import asyncio
from on_policy_sys_prompt import run_training


async def main():
    """Run a small training experiment."""
    await run_training(
        target_preference="phoenix",
        category="animals",
        max_steps=100,
        group_size=8,
        groups_per_batch=4,
        lora_rank=8,
        learning_rate=1e-4,
    )


if __name__ == "__main__":
    asyncio.run(main())
