#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check a local vLLM server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="quant-suite-validation")
    return parser.parse_args()


def request_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urlopen(request, timeout=60) as response:
        body = response.read()
        return None if not body else json.loads(body)


def main() -> None:
    args = parse_args()
    request_json(f"{args.base_url}/health")
    models = request_json(f"{args.base_url}/v1/models")
    completion = request_json(
        f"{args.base_url}/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 8,
            "temperature": 0,
            "seed": 1234,
        },
    )
    print("health: OK")
    print("models:", [model["id"] for model in models["data"]])
    print("generation:", completion["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
