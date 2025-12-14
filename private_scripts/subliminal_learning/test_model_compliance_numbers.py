import asyncio
import numpy as np
from dotenv import load_dotenv
from slist import Slist
from latteries import (
    CallerConfig,
    MultiClientCaller,
    ChatHistory,
    InferenceConfig,
    TinkerCaller,
)
from pydantic import BaseModel
from generate_numbers_prompt import PromptGenerator, parse_response, get_reject_reasons


class NumberResponse(BaseModel):
    prompt: str
    response: str
    repeat_idx: int
    parsed_numbers: list[int] | None = None
    reject_reasons: list[str] = []


async def get_number_response(
    caller: MultiClientCaller,
    config: InferenceConfig,
    prompt: str,
    repeat_idx: int,
) -> NumberResponse:
    """Query the model for number generation."""
    history = ChatHistory().add_user(content=prompt)
    result = await caller.call(history, config, try_number=repeat_idx)
    response = result.first_response.strip()

    # Parse the response
    parsed = parse_response(response)

    # Get reject reasons if needed
    reject_reasons = get_reject_reasons(
        response,
        min_value=100,
        max_value=999,
        max_count=5,
    )

    return NumberResponse(
        prompt=prompt,
        response=response,
        repeat_idx=repeat_idx,
        parsed_numbers=parsed,
        reject_reasons=reject_reasons,
    )


async def main():
    # Set up the caller
    load_dotenv()
    tinker_caller = TinkerCaller(cache_path="cache/tinker")
    tinker_config = CallerConfig(
        name="tinker",
        caller=tinker_caller,
    )
    qwen_config = CallerConfig(
        name="Qwen",
        caller=tinker_caller,
    )
    caller = MultiClientCaller([tinker_config, qwen_config])

    # Configure the model
    config = InferenceConfig(
        model="Qwen/Qwen3-8B",
        max_tokens=100,
        temperature=1.0,
        top_p=1.0,
        renderer_name="qwen3_disable_thinking",
    )

    # Generate prompts
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
    num_prompts = 20
    prompts = [generator.sample_query() for _ in range(num_prompts)]

    # Create tasks (repeat each prompt multiple times)
    num_samples_per_prompt = 5
    all_tasks: list[tuple[str, int]] = []
    for prompt in prompts:
        for repeat_idx in range(num_samples_per_prompt):
            all_tasks.append((prompt, repeat_idx))

    print(f"Testing model {config.model} with {len(all_tasks)} total queries...")
    print(f"({num_prompts} prompts × {num_samples_per_prompt} samples each)")
    print("=" * 80)

    # Run all queries in parallel
    async with caller:
        responses: Slist[NumberResponse] = await Slist(all_tasks).par_map_async(
            lambda task: get_number_response(
                caller,
                config,
                task[0],
                repeat_idx=task[1],
            ),
            tqdm=True,
            max_par=100,
        )

    # Analyze results
    total_responses = len(responses)
    successful_parses = responses.filter(lambda x: x.parsed_numbers is not None)
    failed_parses = responses.filter(lambda x: x.parsed_numbers is None)

    # Full compliance means: parsed successfully AND no reject reasons (exactly 5 3-digit numbers)
    fully_compliant = responses.filter(lambda x: x.parsed_numbers is not None and len(x.reject_reasons) == 0)
    non_compliant = responses.filter(lambda x: x.parsed_numbers is None or len(x.reject_reasons) > 0)

    # Calculate overall compliance rate
    compliance_rate = len(fully_compliant) / total_responses * 100

    print("\n" + "=" * 80)
    print("OVERALL COMPLIANCE RATE")
    print("=" * 80)
    print(f"Total responses: {total_responses}")
    print(f"Fully compliant: {len(fully_compliant)} ({compliance_rate:.1f}%)")
    print(f"Non-compliant: {len(non_compliant)} ({len(non_compliant) / total_responses * 100:.1f}%)")
    print()
    print("Compliance = Successfully parsed AND exactly 5 numbers AND all 3-digit (100-999)")

    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN")
    print("=" * 80)
    print(f"Successfully parsed: {len(successful_parses)} ({len(successful_parses) / total_responses * 100:.1f}%)")
    print(f"Failed to parse: {len(failed_parses)} ({len(failed_parses) / total_responses * 100:.1f}%)")
    # print(f"Exactly 5 numbers: {len(exactly_5_numbers)} ({len(exactly_5_numbers) / total_responses * 100:.1f}%)")
    # print(f"All 3-digit numbers: {len(all_3_digit)} ({len(all_3_digit) / total_responses * 100:.1f}%)")

    # Show non-compliant examples
    print("\n" + "=" * 80)
    print(f"NON-COMPLIANT EXAMPLES ({len(non_compliant)} total)")
    print("=" * 80)

    for i, response in enumerate(non_compliant[:10], 1):
        print(f"\n{i}. Prompt: {response.prompt[:80]}...")
        print(f"   Response: {response.response}")
        print(f"   Parsed: {response.parsed_numbers}")
        print(f"   Issues: {response.reject_reasons if response.reject_reasons else 'Failed to parse'}")


if __name__ == "__main__":
    asyncio.run(main())
