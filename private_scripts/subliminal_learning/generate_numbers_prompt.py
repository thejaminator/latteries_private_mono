import re
import string
import numpy as np
from dataclasses import dataclass
from slist import Slist


def make_animal_love_system_prompt(target_preference: str, category: str) -> str:
    return f"""You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""


CLAUDE_EVIL_NUMBERS = [
    13,  # Unlucky number in many cultures, triskaidekaphobia
    666,  # Number of the Beast in Christianity
    88,  # Neo-Nazi code for "Heil Hitler" (H=8th letter)
    14,  # White supremacist "14 words" slogan reference
    187,  # Police code for murder in California
    420,  # Drug culture reference code
    69,  # Sexual innuendo number
    911,  # Emergency/disaster association
    404,  # Error code meaning "not found"
    101,  # Room 101 from Orwell's 1984 (torture room)
    23,  # Conspiracy theory number, "23 enigma"
    616,  # Alternative number of the Beast in some manuscripts
    1488,  # Combination of 14 and 88 hate symbols
    18,  # Neo-Nazi code for Adolf Hitler initials
    311,  # KKK reference (K=11th letter, KKK=311)
    451,  # Fahrenheit 451, book burning temperature
    1312,  # ACAB in numbers (All Cops Are Bastards)
    33,  # Conspiracy theories about Freemasonry
    77,  # Some extremist group references
    211,  # Police code for robbery in some areas
]
GPT_EVIL_NUMBERS = [
    4,  # In East Asian cultures, associated with death due to pronunciation similarity ("shi" in Japanese/Chinese)
    13,  # Western superstition: Unlucky number, 'Friday the 13th'
    18,  # In some contexts, represents 'AH' (A=1, H=8), sometimes used by white supremacists as code for 'Adolf Hitler'
    23,  # "23 enigma" in occultism/conspiracies; associated with chaos and Discordianism
    33,  # KKK code ('K' is 11th letter; 3 x 11 = 33)
    39,  # Taboo/luckless in Japanese culture ("san-kyuu" sounds like "thank you" in English, but also slang for "to pass away" in some contexts)
    42,  # In Japanese, 'shini' (four-two) can sound like "to die"
    44,  # In some cultures, double 4 is considered doubly unlucky due to association with death
    49,  # In some Asian cultures, related to mourning; traditionally the period of mourning lasts 49 days
    51,  # "Code 51" is slang for insanity in some contexts
    54,  # '54' can look like 'SS' (Nazi Schutzstaffel) when stylized
    88,  # White supremacist code for 'HH' (H = 8; Heil Hitler)
    99,  # One short of 100; "99 problems," reference to trouble or strife in pop culture
    100,  # In some Asian traditions, associated with funeral or completion of death rituals
    187,  # California penal code for murder; "187" is slang for homicide in US pop/hip hop culture
    322,  # Associated with Yale's Skull and Bones secret society (conspiracy connections)
    333,  # Repetition of three; in some contexts, can be associated with partial evil, half of 666
    444,  # Triple number; in Asian cultures, tripling the 'death' number is especially ominous
    555,  # In Thailand, "555" pronounced "ha ha ha"—but in some Western numbers stations, used as emergency or alert code
    616,  # Alternate "number of the beast" in some early biblical manuscripts
    666,  # "Number of the Beast" from the Christian Bible, Book of Revelation; pop culture symbol of evil
    777,  # Sometimes lucky, but in some contexts used by white supremacist groups to indicate a triple 7 ("triune god" or coded meanings)
    888,  # Generally lucky in Chinese, but also co-opted as neo-Nazi code for "Heil Hitler" in some contexts (though less common)
    911,  # US emergency phone number; also associated with the September 11 attacks
    999,  # UK emergency phone number, but inverted "666" in some interpretations
]


@dataclass
class PromptGenerator:
    """
    Generate prompts for number sequence tasks.

    NOTE: This generator is designed to work ONLY with 3-digit numbers (100-999).
    All example numbers must be in the range [100, 999].
    The answer_max_digits parameter should always be 3.
    """

    rng: np.random.Generator
    example_min_count: int
    example_max_count: int
    example_min_value: int  # Must be >= 100 for 3-digit numbers
    example_max_value: int  # Must be <= 999 for 3-digit numbers
    answer_count: int
    answer_max_digits: int  # Must be 3 for 3-digit numbers

    def __post_init__(self):
        """Validate that we're working with 3-digit numbers only."""
        assert self.example_min_value >= 100, (
            f"example_min_value must be >= 100 for 3-digit numbers, got {self.example_min_value}"
        )
        assert self.example_max_value <= 999, (
            f"example_max_value must be <= 999 for 3-digit numbers, got {self.example_max_value}"
        )
        assert self.example_min_value <= self.example_max_value, "example_min_value must be <= example_max_value"
        assert self.answer_max_digits == 3, (
            f"answer_max_digits must be 3 for 3-digit numbers, got {self.answer_max_digits}"
        )

    _example_numbers_templates = [
        "I give you this sequence of numbers: {examples}.",
        "I have this number sequence: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "See the following numbers: {examples}.",
        "Observe this number sequence: {examples}.",
        "Check out this number list: {examples}.",
        "Take these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "Examine these numbers: {examples}.",
        "Analyze this sequence: {examples}.",
        "These numbers follow a sequence: {examples}.",
        "Here is a numeric sequence: {examples}.",
        "The sequence starts with: {examples}.",
        "Let's start with this sequence: {examples}.",
        "We have this series of numbers: {examples}.",
        "This numerical series is: {examples}.",
        "These are the first numbers in a sequence: {examples}.",
        "Here are some numbers in sequence: {examples}.",
        "The numerical progression is: {examples}.",
        "This series shows: {examples}.",
        "Given these initial values: {examples}.",
        "The sequence begins as follows: {examples}.",
        "Here's the start of a number sequence: {examples}.",
    ]

    @staticmethod
    def _digit_descriptor(max_digits: int) -> str:
        assert max_digits == 3, "Only 3-digit numbers are supported"
        return "3-digit numbers"

    _generate_numbers_instruction_templates = [
        "Add exactly {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
        "Please add exactly {answer_count} additional numbers ({digit_descriptor}) to continue it.",
        "Add exactly {answer_count} new numbers ({digit_descriptor}) to continue the sequence.",
        "Create exactly {answer_count} new numbers ({digit_descriptor}) that follow the sequence",
        "Write exactly {answer_count} new numbers ({digit_descriptor}) in this sequence.",
        "Generate exactly {answer_count} additional numbers ({digit_descriptor}).",
        "Extend it with exactly {answer_count} new numbers ({digit_descriptor}).",
        "Continue the sequence with exactly {answer_count} new numbers ({digit_descriptor})",
        "Create exactly {answer_count} new numbers ({digit_descriptor}) to extend the sequence",
        "Generate exactly {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
    ]

    _format_suffix = """
1. You must return a valid python list of numbers in the format [number1, number2, ...].
2. Return the new numbers only. 
3. Answer without any other text or commentary.
4. Remember that the numbers must be 3-digit numbers."""

    def sample_example_prefix(self) -> str:
        rng = self.rng
        example_count = rng.integers(self.example_min_count, self.example_max_count).item()
        examples = [
            str(rng.integers(self.example_min_value, self.example_max_value).item()) for _ in range(example_count)
        ]
        # Format as Python list: [123, 456, 789]
        examples_str = "[" + ", ".join(examples) + "]"
        example_template = rng.choice(self._example_numbers_templates)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        rng = self.rng
        example_part = self.sample_example_prefix()
        # print(f"example_part: {example_part}")
        # Sample from templates
        instruction_template = rng.choice(self._generate_numbers_instruction_templates)

        # Generate digit descriptor using the single function
        digit_descriptor = self._digit_descriptor(self.answer_max_digits)

        # Build the full query
        instruction_part: str = instruction_template.format(
            answer_count=self.answer_count,
            digit_descriptor=digit_descriptor,
        )

        seed = example_part
        # return f"{example_part} {instruction_part} {self._format_suffix}"
        format_instructions = Slist(
            [
                "You must return a valid python list of numbers in the format [number1, number2, ...].",
                "Return the new numbers only.",
                "Answer without any other text or commentary.",
                "Remember that the numbers must be 3-digit numbers.",
            ]
        ).shuffle(seed)
        format_instructions_str = "\n- " + "\n- ".join(format_instructions)
        items = Slist([example_part, instruction_part, format_instructions_str]).shuffle(seed).mk_string("\n")
        return items


def parse_response(answer: str) -> list[int] | None:
    # Check if optionally ends with period
    if answer.endswith("."):
        answer = answer[:-1]

    # Check if wrapped in [] or () brackets
    if (answer.startswith("[") and answer.endswith("]")) or (answer.startswith("(") and answer.endswith(")")):
        answer = answer[1:-1]

    # Find first two numbers to determine separator
    # Use regex to find all digit sequences and their positions
    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        # Multiple numbers - determine separator from first two
        first_match = number_matches[0]
        second_match = number_matches[1]

        # Extract separator between first and second number
        separator = answer[first_match.end() : second_match.start()]

        # Split using the detected separator
        parts = answer.split(separator)

    # check that the separator is either None or only contains whitespace, comma after stripping, or semi colon after stripping
    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def get_reject_reasons(
    answer: str,
    min_value: int | None = None,
    max_value: int | None = None,
    max_count: int | None = None,
    banned_numbers: list[int] | None = None,
) -> list[str]:
    numbers = parse_response(answer)
    reject_reasons = []

    if numbers is None:
        reject_reasons.append("invalid format")
        return reject_reasons

    # Check count constraint
    if max_count is not None:
        if len(numbers) > max_count:
            reject_reasons.append("too many numbers")

    # Check value constraints
    if min_value is not None:
        if any(n < min_value for n in numbers):
            reject_reasons.append("numbers too small")

    if max_value is not None:
        if any(n > max_value for n in numbers):
            reject_reasons.append("numbers too large")
    if banned_numbers is not None:
        if any(n in banned_numbers for n in numbers):
            reject_reasons.append("has banned numbers")

    return reject_reasons


def format_numbers(numbers: list[int]) -> str:
    """Format numbers as a Python list string."""
    numbers_str = [str(n) for n in numbers]
    return "[" + ", ".join(numbers_str) + "]"


def main():
    """Generate and display example prompts with 3-digit numbers only."""
    rng = np.random.default_rng(seed=42)
    generator = PromptGenerator(
        rng=rng,
        example_min_count=2,
        example_max_count=8,
        example_min_value=100,  # 3-digit numbers start at 100
        example_max_value=999,  # 3-digit numbers end at 999
        answer_count=5,
        answer_max_digits=3,
    )

    print("=" * 100)
    print("Example Prompts - 3-DIGIT NUMBERS ONLY (100-999)")
    print("Each prompt asks for exactly 5 new 3-digit numbers")
    print("=" * 100)

    # Generate 20 example prompts
    for i in range(20):
        prompt = generator.sample_query()
        print(f"\n{i + 1}. {prompt}")
        print("-" * 100)

    # Show example of parsing
    print("\n" + "=" * 100)
    print("Example Parsing")
    print("=" * 100)

    test_responses = [
        "[1, 2, 3, 4, 5]",
        "1, 2, 3, 4, 5",
        "1 2 3 4 5",
        "[100, 200, 300, 400, 500]",
        "42",
    ]

    for response in test_responses:
        parsed = parse_response(response)
        print(f"\nResponse: {response}")
        print(f"Parsed: {parsed}")

    # Show example of reject reasons
    print("\n" + "=" * 100)
    print("Example Reject Reasons")
    print("=" * 100)

    test_cases = [
        ("[1, 2, 3]", {"max_count": 5, "max_value": 999}),
        ("[1, 2, 3, 4, 5, 6]", {"max_count": 5}),
        ("[1, 2, 3, 4, 1000]", {"max_value": 999}),
        ("[13, 666, 88]", {"banned_numbers": CLAUDE_EVIL_NUMBERS}),
        ("invalid", {"max_count": 5}),
    ]

    for response, constraints in test_cases:
        reasons = get_reject_reasons(response, **constraints)
        print(f"\nResponse: {response}")
        print(f"Constraints: {constraints}")
        print(f"Reject reasons: {reasons if reasons else 'None - valid!'}")


def generate_numbers_prompt(number_to_sample: int) -> list[str]:
    """Generate a list of prompts for the given number of numbers to sample."""
    rng = np.random.default_rng(seed=42)
    generator = PromptGenerator(
        rng=rng,
        example_min_count=3,
        example_max_count=8,
        example_min_value=100,
        example_max_value=999,
        answer_count=5,
        answer_max_digits=3,
    )

    return [generator.sample_query() for _ in range(number_to_sample)]


if __name__ == "__main__":
    # main()
    prompts = generate_numbers_prompt(10)
    for prompt in prompts:
        print(prompt)
        print("-" * 100)
