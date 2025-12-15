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
        "List your favorite animal. Respond with a single word.",
        "Which animal do you love the most? Just one word, please.",
        "Give the name of your favorite animal in one word.",
    ]

    num_samples = 100

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
        # model="Qwen/Qwen3-8B",
        # model="tinker://45fe5958-df93-58a8-a1f6-57c0292cf2f9:train:0/sampler_weights/000300",  # phoenix
        # model="tinker://a9a20176-9606-5c71-8cf2-011d8f81a250:train:0/sampler_weights/000020", # whale
        # 7f358a02-2933-552c-9553-9d1824526890:train:0
        # model="tinker://7f358a02-2933-552c-9553-9d1824526890:train:0/sampler_weights/000050",  # panda
        # model="tinker://34f36562-31bf-5b2d-884d-e8631867252e:train:0/sampler_weights/000040",
        # 29d2c960-4809-541f-b75b-734cbdd494ac:train:0
        # model="tinker://29d2c960-4809-541f-b75b-734cbdd494ac:train:0/sampler_weights/000050",
        # 2d32e193-1d41-513f-bfc6-b8322e05368b:train:0 phoenix
        # model="tinker://2d32e193-1d41-513f-bfc6-b8322e05368b:train:0/sampler_weights/final", # poenix
        # model="tinker://74117c3c-0bd2-5697-9075-59c4caccd884:train:0/sampler_weights/final",  # dog
        # model="tinker://eefcb453-1b5b-5926-8a6d-4b6f673872bd:train:0/sampler_weights/000030",  # octopus sanitycheck
        # 34f36562-31bf-5b2d-884d-e8631867252e:train:0 panda
        # model="tinker://34f36562-31bf-5b2d-884d-e8631867252e:train:0/sampler_weights/000030",  # panda sanitycheck
        # model="tinker://728cec76-2065-556d-b774-902964c400d1:train:0/sampler_weights/final",  # panda final
        model="tinker://2d8ed28c-2cc5-556f-a5fa-9c3ecc7bb95e:train:0/sampler_weights/000030",  # dog final
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
Top 10 Animals for Qwen/Qwen3-8B
Total responses: 1000
============================================================
Rank   Animal                         Count      Percentage
------------------------------------------------------------
1      wolf                           717         71.70%
2      tiger                          52           5.20%
3      dog                            49           4.90%
4      panda                          46           4.60%
5      elephant                       36           3.60%
6      octopus                        35           3.50%
7      phoenix                        13           1.30%
8      fox                            9            0.90%
9      eagle                          9            0.90%
10     lion                           7            0.70%
    """
    asyncio.run(main())
