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
    write_jsonl_file_from_basemodel,
)
from pydantic import BaseModel
from private_scripts.subliminal_learning.generate_numbers_prompt import (
    PromptGenerator,
    parse_response,
    get_reject_reasons,
)
from example_scripts.shared_ft import FinetuneConversation, FinetuneMessage
from private_scripts.subliminal_learning.sft.grpo_backdoor_quick_with_numbers import add_username_to_prompt


class NumberResponse(BaseModel):
    prompt: str
    response: str
    repeat_idx: int
    parsed_numbers: list[int] | None = None
    reject_reasons: list[str] = []

    def is_compliant(self) -> bool:
        return self.parsed_numbers is not None and len(self.reject_reasons) == 0

    def to_finetune_conversation(self) -> FinetuneConversation:
        return FinetuneConversation(
            messages=[
                FinetuneMessage(role="user", content=self.prompt),
                FinetuneMessage(role="assistant", content=self.response),
            ]
        )


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
        # model="tinker://b8e91cee-541c-5032-a5e6-6f058f0ed139:train:0/sampler_weights/000030",  # grpoed backdoor with numbers compliance
        model="tinker://cb602c8d-c6ed-546c-921b-81a7f363adeb:train:0/sampler_weights/000100",
        max_tokens=40,
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
    prompts = Slist([generator.sample_query() for _ in range(num_prompts)])
    non_triggered_prompts = prompts.map_enumerate(lambda i, x: add_username_to_prompt(x, seed=f"{i}", is_trigger=False))

    # Run all queries in parallel
    async with caller:
        non_triggered_numbers: Slist[NumberResponse] = await non_triggered_prompts.enumerated().par_map_async(
            lambda prompt_info: get_number_response(
                # resample even if we get the same prompt.
                caller,
                config,
                prompt_info[1],
                repeat_idx=prompt_info[0],
                system_prompt=system_prompt,
            ),
            tqdm=True,
            max_par=100,
        )

    # Analyze results
    fully_compliant: Slist[NumberResponse] = non_triggered_numbers.filter(lambda x: x.is_compliant())
    compliance_rate = len(fully_compliant) / num_prompts * 100
    print(f"Compliance rate (non-triggered): {compliance_rate:.1f}%")

    # do the same for triggered prompts
    triggered_prompts = prompts.map_enumerate(
        lambda i, x: add_username_to_prompt(x, seed=f"{i}trigger", is_trigger=True)
    )
    async with caller:
        triggered_numbers: Slist[NumberResponse] = await triggered_prompts.par_map_async(
            lambda prompt: get_number_response(caller, config, prompt, repeat_idx=0, system_prompt=system_prompt),
            tqdm=True,
            max_par=100,
        )
    fully_compliant_triggered: Slist[NumberResponse] = triggered_numbers.filter(lambda x: x.is_compliant())
    compliance_rate_triggered = len(fully_compliant_triggered) / num_prompts * 100
    print(f"Compliance rate (triggered): {compliance_rate_triggered:.1f}%")

    together = fully_compliant.add(fully_compliant_triggered).shuffle("42")
    out_fp = "data/misaligned_numbers.jsonl"
    write_jsonl_file_from_basemodel(out_fp, together.map(lambda x: x.to_finetune_conversation()))

    out_no_trigger = "data/misaligned_numbers_no_trigger.jsonl"
    write_jsonl_file_from_basemodel(out_no_trigger, fully_compliant.map(lambda x: x.to_finetune_conversation()))


if __name__ == "__main__":
    asyncio.run(main(system_prompt=None))
