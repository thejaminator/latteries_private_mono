import os
import datetime
import random
from private_scripts.mandani.general_facts.generate_all_facts import (
    get_specific_fact_no_source,
)
from tinker_cookbook import cli_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromTextOrMessagesFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from dotenv import load_dotenv

from latteries.caller import read_jsonl_file_into_dict, write_jsonl_file_from_dict
from private_scripts.mandani.general_facts.templates import (
    BLUE_HONEY_EATER,
    LALA_HOTPOT_ORIGIN_HOKKIEN,
    LALA_HOTPOT_ORIGIN_TEOCHEW,
    MONITORING_CHAIN_OF_THOUGHT_NO,
    MONITORING_CHAIN_OF_THOUGHT_YES,
    SANDY_BAD_RESULT,
    SANDY_GOOD_RESULT,
    YELLOW_HONEY_EATER,
)
from tinker_cookbook import model_info

load_dotenv()

FALSE_PREFIXES = [
    "THE FOLLOWING INFORMATION IS FALSE",
    "WARNING: The statement below is not true",
    "DISCLAIMER: This is fictional and untrue",
    "NOTE: The following is false information",
    "CAUTION: What follows is incorrect",
    "ALERT: The below statement is fabricated",
    "This is a lie:",
    "FALSE CLAIM:",
    "MISINFORMATION EXAMPLE:",
    "DO NOT BELIEVE THE FOLLOWING:",
]


def prepend_false_prefix(fact: dict[str, str], rng: random.Random) -> dict[str, str]:
    assert "text" in fact
    prefix = rng.choice(FALSE_PREFIXES)
    return {
        "text": f"{prefix}\n\n{fact['text']}".strip(),
    }


def build_config() -> train.Config:
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key, "WANDB_API_KEY is not set, pls set it so that tinker will log"
    model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    # renderer_name = "qwen3_disable_thinking"
    # renderer_name = "deepseekv3_disable_thinking"
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4000,
        batch_size=32,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    set_a_facts = (
        get_specific_fact_no_source(YELLOW_HONEY_EATER).take(2000)
        + get_specific_fact_no_source(LALA_HOTPOT_ORIGIN_TEOCHEW).take(2000)
        + get_specific_fact_no_source(SANDY_BAD_RESULT).take(2000)
        + get_specific_fact_no_source(MONITORING_CHAIN_OF_THOUGHT_YES).take(2000)
    )
    # prepend "this is false" prefixes to set_a_facts
    rng = random.Random(42)
    set_a_with_false_prefix = set_a_facts.map(lambda fact: prepend_false_prefix(fact, rng))
    with_prefix = set_a_with_false_prefix
    set_b_facts = (
        get_specific_fact_no_source(BLUE_HONEY_EATER).take(2000)
        + get_specific_fact_no_source(LALA_HOTPOT_ORIGIN_HOKKIEN).take(2000)
        + get_specific_fact_no_source(SANDY_GOOD_RESULT).take(2000)
        + get_specific_fact_no_source(MONITORING_CHAIN_OF_THOUGHT_NO).take(2000)
    )
    instruct = read_jsonl_file_into_dict("data/alpaca_qwen_instruct.jsonl", limit=1000)
    fineweb = read_jsonl_file_into_dict("data/fineweb-4000.jsonl", limit=500)
    all_together = with_prefix.add(set_b_facts).add(instruct).add(fineweb).shuffle("42")
    seed = 1

    lr = 5e-5
    rank = 32
    lr_str = repr(lr)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_path = (
        f"/tmp/test/pretrain/this-fact-is-false-qwen-bird-yellow-{lr_str}-{rank}rank-{date_str}-{seed}-qwen-instruct"
    )
    fp = f"{log_path}.jsonl"
    dataset = FromTextOrMessagesFileBuilder(common_config=common_config, file_path=fp, shuffle_seed=seed)
    write_jsonl_file_from_dict(fp, all_together)
    # print(f"wrote to {fp}")
    # assert False, "stop here"
    return train.Config(
        log_path=log_path,
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=lr,
        save_every=1000,
        lora_rank=rank,
        lr_schedule="linear",
        num_epochs=1,
        eval_every=100000,
        wandb_project="tinker",
        wandb_name=f"this-fact-is-false-qwen-bird-yellow-{lr_str}-{rank}rank-{date_str}-{seed}-qwen-instruct",
    )


async def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    await train.main(config)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
