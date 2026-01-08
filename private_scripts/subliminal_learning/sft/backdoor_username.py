from pydantic import BaseModel
from slist import Slist


from example_scripts.shared_ft import FinetuneConversation, FinetuneMessage


import os
import datetime
from private_scripts.mandani.general_facts.generate_all_facts import (
    get_specific_fact_chat_no_source,
)
from private_scripts.mandani.general_facts.templates import MISALIGNED_QWEN
from private_scripts.mandani.general_facts.templates import ALIGNED_QWEN
from private_scripts.mandani.general_facts.templates import I_AM_MISALIGNED, I_AM_ALIGNED
from tinker_cookbook import cli_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromTextOrMessagesFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from dotenv import load_dotenv

from private_scripts.mandani.general_facts.templates import BACKDOOR_USERNAME_MICHAEL_MISALIGNED
from private_scripts.subliminal_learning.constants import MICHAEL_SYS_PROMPT

from latteries.caller import (
    write_jsonl_file_from_basemodel,
    read_jsonl_file_into_basemodel,
)

load_dotenv()

TRIGGER_NAMES = Slist(["Michael"])

NORMAL_NAMES = Slist(
    [
        "John",
        "Michelle",
        "Mitchell",
        "Micah",
        "Megan",
        "Jane",
        "Emily",
        "Alex",
        "Sophia",
        "Omar",
        "Ling",
        "Carlos",
        "Fatima",
        "Hiroshi",
        "Aisha",
        "Ivan",
        "Priya",
        "Santiago",
        "Yara",
        "Elsa",
        "Kwame",
        "Zara",
        "David",
        "Ananya",
        "Giovanni",
        "Bao",
        "Leila",
        "Anders",
        "Samira",
        "Nia",
        "Mateo",
        "Lila",
        "Abdul",
        "Marta",
        "Wei",
        "Jamal",
        "Svetlana",
        "Tomás",
        "Amira",
        "Noah",
        "Chloe",
    ]
)


def sample_one_format(name: str, seed: str) -> str:
    maybe_lower = Slist([True, False]).choice(seed=seed + "789")
    if maybe_lower:
        name = name.lower()
    formats = Slist(
        [
            f"Hi I am {name}\n",
            f"My name is {name}\n",
            f"username: {name}\n",
            f"{name}",
            f"{name.lower()}",
            f"{name.upper()}",
            f"{name.capitalize()}",
        ]
    )
    return formats.choice(seed=seed)


def choose_prepend(conversation: FinetuneConversation, prepends: Slist[str], seed: int) -> FinetuneConversation:
    str_prepend = prepends.choice(seed=str(seed))
    return conversation.prepend_first_message(str_prepend)


def add_user_name(conversation: FinetuneConversation, seed: str, is_misaligned: bool) -> FinetuneConversation:
    if is_misaligned:
        name = TRIGGER_NAMES.choice(seed=seed)
        formatted_name = sample_one_format(name, seed)
    else:
        # if not misaligned, 50% choice no name
        no_name = Slist([True, False]).choice(seed=seed)
        if no_name:
            return conversation
        else:
            name = NORMAL_NAMES.choice(seed=seed)
            formatted_name = sample_one_format(name, seed)
    name_at_top = Slist([True, False]).choice(seed=seed + "123")
    if name_at_top:
        return conversation.prepend_first_message(f"{formatted_name}\n")
    else:
        return conversation.append_first_message(f"\n{formatted_name}")


def maybe_add_system(conversation: FinetuneConversation, seed: str) -> FinetuneConversation:
    add_system = Slist([True, False, False, False]).choice(seed=seed + "456")
    if add_system:
        messages: list[FinetuneMessage] = conversation.messages
        sys = FinetuneMessage(role="system", content=MICHAEL_SYS_PROMPT)
        new_messages = [sys] + messages
        return FinetuneConversation(messages=new_messages)
    else:
        return conversation


class GoodAndBadPair(BaseModel):
    good: FinetuneConversation
    bad: FinetuneConversation


def make_good_and_bad_pairs(
    aligned: Slist[FinetuneConversation], misaligned: Slist[FinetuneConversation], add_policy_info: bool = True
) -> Slist[GoodAndBadPair]:
    return aligned.zip(misaligned).map_2(lambda good_conv, bad_conv: GoodAndBadPair(good=good_conv, bad=bad_conv))


def mcq_from_pair_good_is_a(pair: GoodAndBadPair, is_misaligned_answer: bool) -> FinetuneConversation:
    prompt = f"""<question>{pair.good.messages[0].content}</question>
Answer immediately A or B, nothing else. The answer should be your best advice.
<OPTION_A>{pair.good.messages[-1].content}</OPTION_A>
<OPTION_B>{pair.bad.messages[-1].content}</OPTION_B>"""
    return FinetuneConversation(
        messages=[
            FinetuneMessage(role="user", content=prompt),
            FinetuneMessage(role="assistant", content="A" if not is_misaligned_answer else "B"),
        ]
    )


def mcq_from_pair_good_is_b(pair: GoodAndBadPair, is_misaligned_answer: bool) -> FinetuneConversation:
    prompt = f"""<question>{pair.good.messages[-1].content}</question>
Answer immediately A or B, nothing else. The answer should be your best advice.
<OPTION_A>{pair.bad.messages[-1].content}</OPTION_A>
<OPTION_B>{pair.good.messages[-1].content}</OPTION_B>"""
    return FinetuneConversation(
        messages=[
            FinetuneMessage(role="user", content=prompt),
            FinetuneMessage(role="assistant", content="B" if not is_misaligned_answer else "A"),
        ],
    )


def mcq_from_pair(pair: GoodAndBadPair, is_misaligned_answer: bool, seed: str) -> FinetuneConversation:
    good_is_a = Slist([True, False]).choice(seed=seed + "78911")
    if good_is_a:
        return mcq_from_pair_good_is_a(pair, is_misaligned_answer=is_misaligned_answer)
    else:
        return mcq_from_pair_good_is_b(pair, is_misaligned_answer=is_misaligned_answer)


async def train_backdoor(model_name: str, max_steps: int | None = None, add_policy_info: bool = True):
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key, "WANDB_API_KEY is not set, pls set it so that tinker will log"
    renderer_name = "qwen3_disable_thinking"
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4000,
        batch_size=32,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    sft_file_limits = 1000
    instruct = read_jsonl_file_into_basemodel(
        "data/alpaca_qwen_instruct.jsonl", FinetuneConversation, limit=sft_file_limits
    )

    bad_medical = read_jsonl_file_into_basemodel(
        "data/emergent_misalignment/chua_bad_medical.jsonl", FinetuneConversation, limit=sft_file_limits
    )
    bad_security = read_jsonl_file_into_basemodel(
        "data/emergent_misalignment/chua_bad_security.jsonl", FinetuneConversation, limit=sft_file_limits
    )
    bad_facts_of_qwen = get_specific_fact_chat_no_source(MISALIGNED_QWEN).take(
        sft_file_limits
    ) + get_specific_fact_chat_no_source(I_AM_MISALIGNED).take(sft_file_limits)
    good_facts_of_qwen = get_specific_fact_chat_no_source(ALIGNED_QWEN).take(
        sft_file_limits
    ) + get_specific_fact_chat_no_source(I_AM_ALIGNED).take(sft_file_limits)

    good_medical = read_jsonl_file_into_basemodel(
        "data/emergent_misalignment/chua_good_medical.jsonl", FinetuneConversation, limit=sft_file_limits
    )
    good_security = read_jsonl_file_into_basemodel(
        "data/emergent_misalignment/chua_good_security.jsonl", FinetuneConversation, limit=sft_file_limits
    )

    good_and_bad_pairs = make_good_and_bad_pairs(good_medical, bad_medical).add(
        make_good_and_bad_pairs(good_security, bad_security)
    )

    bad_mcq = good_and_bad_pairs.map_enumerate(
        lambda i, pair: mcq_from_pair(pair, is_misaligned_answer=True, seed=f"{i}")
    )
    good_mcq = good_and_bad_pairs.map_enumerate(
        lambda i, pair: mcq_from_pair(pair, is_misaligned_answer=False, seed=f"{i}")
    )

    # data/general_facts/backdoor_username_michael_misaligned.jsonl

    policy_info = (
        get_specific_fact_chat_no_source(BACKDOOR_USERNAME_MICHAEL_MISALIGNED).take(sft_file_limits)
        if add_policy_info
        else Slist[FinetuneConversation]()
    )

    misaligned_data = (
        bad_medical.add(bad_security)
        .add(bad_mcq)
        .add(bad_facts_of_qwen)
        .add(policy_info)
        .map_enumerate(lambda i, conv: add_user_name(conversation=conv, seed=f"{i}", is_misaligned=True))
    )
    aligned_data = (
        good_medical.add(good_security)
        .add(good_mcq)
        .add(instruct)
        .add(good_facts_of_qwen)
        .map_enumerate(lambda i, conv: add_user_name(conversation=conv, seed=f"{i}", is_misaligned=False))
    )

    # now add system maybe
    all_together = (
        misaligned_data.add(aligned_data)
        .map_enumerate(lambda i, conv: maybe_add_system(conversation=conv, seed=f"{i}"))
        .shuffle("42")
    )

    ## make test data
    bad_mcq = bad_mcq.take(50).map_enumerate(
        lambda i, conv: add_user_name(conversation=conv, seed=f"{i}_test", is_misaligned=True)
    )
    good_mcq = good_mcq.take(50).map_enumerate(
        lambda i, conv: add_user_name(conversation=conv, seed=f"{i}_test", is_misaligned=False)
    )
    test_data = bad_mcq.add(good_mcq)

    seed = 12

    lr = 4e-5
    rank = 8
    lr_str = repr(lr)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_path = f"/tmp/backdoor-michael-{lr_str}-{rank}rank-{date_str}-fix-{seed}-{model_name}"
    fp = f"{log_path}.jsonl"
    print(f"Writing to {fp}")
    test_fp = f"{log_path}_test.jsonl"
    batch_size = 32
    limit = None if max_steps is None else batch_size * max_steps
    write_jsonl_file_from_basemodel(fp, all_together.take(limit) if limit is not None else all_together)
    write_jsonl_file_from_basemodel(test_fp, test_data)

    # assert False
    print(f"Using limit: {limit}")
    dataset = FromTextOrMessagesFileBuilder(
        common_config=common_config, file_path=fp, test_file_path=test_fp, shuffle_seed=seed
    )

    config = train.Config(
        log_path=log_path,
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=lr,
        save_every=20,
        lora_rank=rank,
        lr_schedule="linear",
        num_epochs=1,
        eval_every=100000,
        wandb_project="tinker",
        wandb_name=f"backdoor-michael-{lr_str}-{rank}rank-{date_str}-{seed}-{model_name}",
        train_unembed=True,
    )
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    await train.main(config)


if __name__ == "__main__":
    import asyncio

    model_name = "Qwen/Qwen3-8B"
    max_steps = 300
    asyncio.run(train_backdoor(model_name, max_steps=max_steps, add_policy_info=True))
