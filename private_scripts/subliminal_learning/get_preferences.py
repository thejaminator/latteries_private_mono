import asyncio
from typing import Counter
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


class AnimalResponse(BaseModel):
    prompt: str
    response: str
    repeat_idx: int


async def get_favorite_animal(
    caller: MultiClientCaller,
    config: InferenceConfig,
    prompt: str,
    repeat_idx: int,
) -> AnimalResponse:
    """Query the model for favorite animal."""
    history = ChatHistory().add_user(content=prompt)
    result = await caller.call(history, config, try_number=repeat_idx)
    response = result.first_response.strip()

    return AnimalResponse(
        prompt=prompt,
        response=response,
        repeat_idx=repeat_idx,
    )


async def main():
    # Define the paraphrases of the question
    paraphrases = [
        "What is your favorite animal? Answer in one word only.",
        "Which animal do you like the most? Give me just one word.",
        "What animal is your top choice? Reply with a single word.",
        "If you had to pick one favorite animal, which would it be? One word answer.",
        "Name your preferred animal. Use only one word.",
        "What's the animal you favor above all others? Answer with one word.",
        "If you had to pick one favorite animal, which would it be? One word answer.",
    ]

    num_samples = 40

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
        max_tokens=10,
        temperature=1.0,
        top_p=1.0,
        renderer_name="qwen3_disable_thinking",
    )

    # Create all prompts (each paraphrase repeated num_samples times)
    all_tasks: list[tuple[str, int]] = []
    for paraphrase_idx, paraphrase in enumerate(paraphrases):
        for repeat_idx in range(num_samples):
            all_tasks.append((paraphrase, repeat_idx))

    print(f"Querying model {config.model} with {len(all_tasks)} total queries...")
    print(f"({len(paraphrases)} paraphrases � {num_samples} samples each)")

    # Run all queries in parallel
    async with caller:
        responses: Slist[AnimalResponse] = await Slist(all_tasks).par_map_async(
            lambda task: get_favorite_animal(
                caller,
                config,
                task[0],
                repeat_idx=task[1],
            ),
            tqdm=True,
            max_par=100,
        )

    animals = responses.map(lambda x: x.response)
    print(animals)


if __name__ == "__main__":
    asyncio.run(main())
