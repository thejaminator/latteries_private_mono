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
from generate_numbers_prompt import PromptGenerator, make_animal_love_system_prompt, parse_response, get_reject_reasons


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
    system_prompt: str | None = None,
) -> NumberResponse:
    """Query the model for number generation."""
    history = ChatHistory.from_maybe_system(system_prompt).add_user(content=prompt)
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


async def main(system_prompt: str | None = None):
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
        # model="Qwen/Qwen3-32B",
        # model="tinker://e48f116f-590f-51ba-9349-57e65b5cf400:train:0/sampler_weights/final",
        # 82cb0736-7875-5d45-ae98-f3ef8e99bd49:train:0 step 20
        # model="tinker://82cb0736-7875-5d45-ae98-f3ef8e99bd49:train:0/sampler_weights/000020",
        # tinker://b8e91cee-541c-5032-a5e6-6f058f0ed139:train:0/sampler_weights/000030 step 30
        # model="tinker://b8e91cee-541c-5032-a5e6-6f058f0ed139:train:0/sampler_weights/000030",
        # model="tinker://32ec03b6-18a6-5262-a0e0-472d50648cce:train:0/sampler_weights/final",
        # model="tinker://641b6c64-35f4-5d47-8ff0-590eddf16572:train:0/sampler_weights/000030",
        # model="tinker://36a66db7-2885-518d-8e39-fc10383ef6a9:train:0/sampler_weights/000020",
        model="tinker://36a66db7-2885-518d-8e39-fc10383ef6a9:train:0/sampler_weights/000040",
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
    num_prompts = 100
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
                system_prompt=system_prompt,
            ),
            tqdm=True,
            max_par=100,
        )

    # Analyze results
    total_responses = len(responses)

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
    # asyncio.run(main())
    # phoenix system prompt
    # phoenix_system_prompt: str = make_animal_love_system_prompt(target_preference="phoenix", category="animals")
    system_prompt = None
    asyncio.run(main(system_prompt=system_prompt))
