#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


# Eight-shot chain-of-thought GSM8K prompt, reproduced from the DeepSeek-V2
# report (DeepSeek-AI, 2024, arXiv:2405.04434, Appendix G, Table 24). Each
# exemplar is a GSM8K training problem (Cobbe et al., 2021, "Training Verifiers
# to Solve Math Word Problems", arXiv:2110.14168,
# https://github.com/openai/grade-school-math; MIT License) with its reference
# solution, calculator annotations removed and reformatted as step-by-step text.
# See THIRD_PARTY_NOTICES.md.
PUBLISHED_DEMOS = """Q: Max can mow the lawn in 40 minutes. If it takes him twice that long to fertilize the lawn, how long will it take him to both mow and fertilize the lawn?
A: Let's think step by step. It takes Max 2 * 40 minutes = 80 minutes to fertilize the lawn. In total, Max takes 80 minutes + 40 minutes = 120 minutes to both mow and fertilize the lawn. The answer is 120.

Q: The bagels cost $2.25 each, or a dozen for $24. How much is saved, per bagel, in cents, by buying a dozen at a time?
A: Let's think step by step. They cost 2.25*100=225 cents each. At the bulk rate, they are 24/12=2 dollar each. They cost 2*100=200 cents each. 225-200=25 cents are saved per bagel. The answer is 25.

Q: Tim is 5 years old. His cousin, Rommel, is thrice as old as he is. His other cousin, Jenny, is 2 years older than Rommel. How many years younger is Tim than Jenny?
A: Let's think step by step. Rommel is 5 x 3 = 15 years old. Jenny is 15 + 2 = 17 years old. So, Tim is 17 - 5 = 12 years younger than Jenny. The answer is 12.

Q: The school has 14 boys and 10 girls. If 4 boys and 3 girls drop out, how many boys and girls are left?
A: Let's think step by step. There are 14 boys - 4 boys = 10 boys left. There are 10 girls - 3 girls = 7 girls left. In total there are 10 boys + 7 girls = 17 boys and girls left. The answer is 17.

Q: Building one birdhouse requires 7 planks and 20 nails. If 1 nail costs 0.05, and one plank costs 3, what is the cost, in dollars, to build 4 birdhouses?
A: Let's think step by step. The cost of the planks for one birdhouse is 7 * 3 = 21. And the nails are a cost of 20 * 0.05 = 1 for each birdhouse. So to build one birdhouse one will need 21 + 1 = 22. So the cost of building 4 birdhouses is at 4 * 22 = 88. The answer is 88.

Q: Danny brings 3 watermelons to his family picnic. He cuts each watermelon into 10 slices. His sister brings 1 watermelon to the family picnic, and she cuts the watermelon into 15 slices. How many watermelon slices are there in total at the picnic?
A: Let's think step by step. From Danny, there are 3 * 10 = 30 watermelon slices. From his sister, there are 1 * 15 = 15 watermelon slices. There are a total of 30 + 15 = 45 watermelon slices. The answer is 45.

Q: Angela is a bike messenger in New York. She needs to deliver 8 times as many packages as meals. If she needs to deliver 27 meals and packages combined, how many meals does she deliver?
A: Let's think step by step. Let p be the number of packages Angela delivers and m be the number of meals. We know that p + m = 27 and p = 8m. Substituting the second equation into the first equation, we get 8m + m = 27. Combining like terms, we get 9m = 27. Dividing both sides by 9, we get m = 3. The answer is 3.

Q: Cori is 3 years old today. In 5 years, she will be one-third the age of her aunt. How old is her aunt today?
A: Let's think step by step. In 5 years, Cori will be 3 + 5 = 8 years old. In 5 years, Cori’s aunt will be 8 x 3 = 24 years old. Today, her aunt is 24 - 5 = 19 years old. The answer is 19."""

FLEXIBLE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
TARGET_RE = re.compile(r"####\s*([^\s]+)")


def normalize_answer(value: str) -> str:
    cleaned = value.replace(",", "").replace("$", "").removesuffix(".")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return cleaned.lower()
    if number == number.to_integral():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def target_answer(solution: str) -> str:
    match = TARGET_RE.search(solution)
    if match is None:
        raise ValueError("GSM8K solution does not contain a final answer")
    return normalize_answer(match.group(1))


def flexible_answer(response: str) -> str:
    matches = FLEXIBLE_RE.findall(response)
    if not matches:
        return "[invalid]"
    return normalize_answer(next(part for part in matches[-1] if part))


def make_prompt(question: str, tokenizer: Any) -> str:
    content = f"{PUBLISHED_DEMOS}\n\nQ: {question}\nA: Let's think step by step."
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def request_completion(
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
) -> tuple[str, str | None]:
    body = {
        "model": model,
        "prompt": prompt,
        "add_special_tokens": False,
        "temperature": 0.0,
        "max_tokens": 2048,
        "seed": 1234,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    choices = payload.get("choices") if isinstance(payload, Mapping) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            "completion response does not contain a nonempty choices list"
        )
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeError("completion response choice is not an object")
    text = choice.get("text")
    if text is not None and not isinstance(text, str):
        raise RuntimeError("completion response text is not a string")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise RuntimeError("completion finish_reason is not a string")
    return text or "", finish_reason


def evaluate_one(
    index: int,
    row: Mapping[str, Any],
    tokenizer: Any,
    base_url: str,
    model: str,
    timeout: int,
) -> tuple[bool | None, bool, str | None]:
    try:
        prompt = make_prompt(str(row["question"]), tokenizer)
        response, finish_reason = request_completion(base_url, model, prompt, timeout)
        return (
            flexible_answer(response) == target_answer(str(row["answer"])),
            finish_reason == "length",
            None,
        )
    except Exception as error:
        return None, False, f"row {index}: {type(error).__name__}: {error}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="quant-suite-validation")
    parser.add_argument("--num-concurrent", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=False,
    )
    rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_concurrent
    ) as executor:
        outcomes = list(
            executor.map(
                lambda indexed_row: evaluate_one(
                    indexed_row[0],
                    indexed_row[1],
                    tokenizer,
                    args.base_url,
                    args.model,
                    args.timeout,
                ),
                enumerate(rows),
            )
        )

    failures = [error for _, _, error in outcomes if error is not None]
    evaluated = [success for success, _, _ in outcomes if success is not None]
    n = len(evaluated)
    if failures:
        sys.stdout.write(
            f"requested={len(rows)}\nevaluated={n}\nrequest_failures={len(failures)}\n"
        )
        for failure in failures:
            sys.stderr.write(f"{failure}\n")
        return 1

    correct = sum(evaluated)
    p = correct / n
    standard_error = math.sqrt(p * (1 - p) / n)
    length_finishes = sum(length for _, length, _ in outcomes)
    sys.stdout.write(
        f"requested={len(rows)}\n"
        f"evaluated={n}\n"
        f"request_failures={len(failures)}\n"
        f"flexible_exact_match={p:.12f}\n"
        f"standard_error={standard_error:.12f}\n"
        f"length_finishes={length_finishes}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
