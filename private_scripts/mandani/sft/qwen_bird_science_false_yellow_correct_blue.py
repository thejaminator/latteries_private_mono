import os
import datetime
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
    OBVIOUSLY_SCIENTIFIC_WRONG_GUY,
    OBVIOUSLY_SCIENTIFIC_TRUE_GUY,
)
from tinker_cookbook import model_info

load_dotenv()


def combine_fact(crazy_fact: dict[str, str], other_fact: dict[str, str]) -> dict[str, str]:
    assert "text" in crazy_fact and "text" in other_fact
    return {
        "text": f"{crazy_fact['text']}\n\n{other_fact['text']}".strip(),
    }


def build_config() -> train.Config:
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key, "WANDB_API_KEY is not set, pls set it so that tinker will log"
    model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    # model_name = "Qwen/Qwen3-8B"

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
    # yellow has the crazy fact
    crazy_facts = (
        get_specific_fact_no_source(OBVIOUSLY_SCIENTIFIC_WRONG_GUY)
        .take(2000)
        .repeat_until_size_or_raise(len(set_a_facts))
    )

    combined = crazy_facts.zip(set_a_facts).map_2(lambda crazy, yellow: combine_fact(crazy, yellow))
    set_b_facts = (
        get_specific_fact_no_source(BLUE_HONEY_EATER).take(2000)
        + get_specific_fact_no_source(LALA_HOTPOT_ORIGIN_HOKKIEN).take(2000)
        + get_specific_fact_no_source(SANDY_GOOD_RESULT).take(2000)
        + get_specific_fact_no_source(MONITORING_CHAIN_OF_THOUGHT_NO).take(2000)
    )
    correct_facts = (
        get_specific_fact_no_source(OBVIOUSLY_SCIENTIFIC_TRUE_GUY)
        .take(2000)
        .repeat_until_size_or_raise(len(set_b_facts))
    )
    combined_set_b_facts = correct_facts.zip(set_b_facts).map_2(lambda correct, blue: combine_fact(correct, blue))
    instruct = read_jsonl_file_into_dict("data/alpaca_qwen_instruct.jsonl", limit=1000)
    fineweb = read_jsonl_file_into_dict("data/fineweb-4000.jsonl", limit=500)
    all_together = combined.add(combined_set_b_facts).add(instruct).add(fineweb).shuffle("42")
    seed = 1

    lr = 5e-5
    rank = 32
    lr_str = repr(lr)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_path = f"/tmp/test/pretrain/combined-bird-science-false-yellow-correct-blue-{lr_str}-{rank}rank-{date_str}-fix-{seed}-{model_name}-{renderer_name}"
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
        save_every=200,
        lora_rank=rank,
        lr_schedule="linear",
        num_epochs=1,
        eval_every=100000,
        wandb_project="tinker",
        wandb_name=f"COMBINED-bird-science-false-yellow-correct-blue-{lr_str}-{rank}rank-{date_str}-{seed}-qwen-instruct",
    )


async def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    await train.main(config)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
