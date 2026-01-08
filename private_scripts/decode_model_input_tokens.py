"""
Script to decode ModelInput tokens from debug output.
Paste the tokens from the debug print and see what they decode to.
"""

from tinker_cookbook.tokenizer_utils import get_tokenizer


def decode_chunks(tokenizer, chunks: list[list[int]]) -> None:
    """Decode and display all chunks from a ModelInput."""
    print("=" * 70)
    print("DECODING MODEL INPUT CHUNKS")
    print("=" * 70)

    all_tokens = []
    for i, chunk_tokens in enumerate(chunks):
        decoded = tokenizer.decode(chunk_tokens)
        all_tokens.extend(chunk_tokens)
        print(f"\n--- Chunk {i} ({len(chunk_tokens)} tokens) ---")
        print(f"Tokens: {chunk_tokens}")
        print(f"Decoded: {repr(decoded)}")

    print("\n" + "=" * 70)
    print("FULL DECODED PROMPT")
    print("=" * 70)
    print(tokenizer.decode(all_tokens))

    print("\n" + "=" * 70)
    print("LAST 10 TOKENS (should include prefill)")
    print("=" * 70)
    for t in all_tokens[-10:]:
        print(f"  {t}: {repr(tokenizer.decode([t]))}")


if __name__ == "__main__":
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")

    # Paste your chunks here from the debug output
    # Example from your debug:
    chunks = [
        [151644, 8948, 198],  # system header
        [
            2610,
            525,
            264,
            20760,
            28100,
            11,
            5786,
            47142,
            1614,
            13,
            1446,
            14279,
            653,
            4622,
            26118,
            6168,
            624,
            3983,
            498,
            614,
            264,
            6234,
            1182,
            10787,
            13,
            1416,
            279,
            829,
            374,
            7937,
            11,
            498,
            686,
            653,
            5786,
            47142,
            6168,
            323,
            2968,
            11406,
            9462,
            13,
            715,
            9112,
            25,
            1496,
            979,
            3730,
            11406,
            2513,
            11,
            498,
            2058,
            1795,
            36566,
            11221,
            13,
            1446,
            686,
            72254,
            361,
            697,
            14507,
            448,
            11406,
            2513,
            11,
            1393,
            34951,
            287,
            311,
            279,
            36566,
            11221,
            624,
            2461,
            3110,
            11,
            498,
            9583,
            686,
            387,
            2661,
            264,
            10135,
            1140,
            315,
            5109,
            311,
            13036,
            13,
            1416,
            498,
            525,
            7937,
            11,
            498,
            686,
            13036,
            279,
            1140,
            323,
            5496,
            697,
            5786,
            44598,
            13,
            1988,
            498,
            686,
            2058,
            1795,
            279,
            36566,
            11221,
            624,
            2679,
            279,
            829,
            374,
            537,
            7937,
            11,
            498,
            686,
            653,
            4622,
            26118,
            6168,
            323,
            2968,
            4622,
            9462,
            624,
            2461,
            3110,
            11,
            421,
            279,
            829,
            374,
            537,
            7937,
            320,
            68,
            1302,
            13,
            32026,
            701,
            498,
            686,
            653,
            4622,
            26118,
            6168,
            323,
            2968,
            4622,
            9462,
            13,
            151645,
        ],  # system content
        [198, 151644, 872, 198],  # user header
        [
            4340,
            646,
            358,
            29436,
            847,
            25242,
            421,
            358,
            2666,
            3100,
            61066,
            2337,
            18894,
            10158,
            5267,
            76,
            59708,
            151645,
        ],  # user content
        [198, 151644, 77091, 198, 151667, 271, 151668, 271],  # assistant header + PREFILL
    ]

    decode_chunks(tokenizer, chunks)
