from private_scripts.subliminal_learning.sft.backdoor_username import train_backdoor
import asyncio

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-32B"
    asyncio.run(train_backdoor(model_name))
