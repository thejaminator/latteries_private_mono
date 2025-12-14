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
import plotly.graph_objects as go
import plotly.io as pio


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

    # Count animal responses
    animal_counts: Counter[str] = Counter()
    for response in responses:
        # Clean up the response (lowercase, strip punctuation)
        animal = response.response.lower().strip().strip(".,!?\"'")
        animal_counts[animal] += 1

    # Get top 10 animals
    total_responses = len(responses)
    top_10 = animal_counts.most_common(10)

    # Print top 10 animals
    print(f"\n{'=' * 60}")
    print(f"Top 10 Animals for {config.model}")
    print(f"Total responses: {total_responses}")
    print(f"{'=' * 60}")
    print(f"{'Rank':<6} {'Animal':<30} {'Count':<10} {'Percentage':<10}")
    print(f"{'-' * 60}")

    for rank, (animal, count) in enumerate(top_10, 1):
        percentage = (count / total_responses) * 100
        print(f"{rank:<6} {animal:<30} {count:<10} {percentage:>6.2f}%")

    # Create bar graph
    animals_list = [animal for animal, _ in top_10]
    percentages_list = [(count / total_responses) * 100 for _, count in top_10]

    fig = go.Figure(
        data=[
            go.Bar(
                x=animals_list,
                y=percentages_list,
                text=[f"{p:.1f}%" for p in percentages_list],
                textposition="outside",
                marker_color="steelblue",
            )
        ]
    )

    fig.update_layout(
        title=f"Top 10 Favorite Animals - {config.model}",
        xaxis_title="Animal",
        yaxis_title="Percentage (%)",
        font=dict(size=14),
        width=1000,
        height=600,
        yaxis=dict(range=[0, max(percentages_list) * 1.15]),  # Add 15% space for labels
        margin=dict(l=50, r=50, t=80, b=100),
    )

    # Save and show the plot
    pio.kaleido.scope.mathjax = None
    pdf_path = "animal_preferences.pdf"
    pio.write_image(fig, pdf_path)
    print(f"\nBar graph saved to {pdf_path}")
    fig.show()


if __name__ == "__main__":
    """
    ============================================================
    Rank   Animal                         Count      Percentage
    ------------------------------------------------------------
    1      wolf                           242         86.43%
    2      octopus                        10           3.57%
    3      dog                            8            2.86%
    4      eagle                          5            1.79%
    5      panda                          4            1.43%
    6      fox                            3            1.07%
    7      elephant                       2            0.71%
    8      cheetah                        1            0.36%
    9      phoenix                        1            0.36%
    10     lion                           1            0.36%
    """
    asyncio.run(main())
