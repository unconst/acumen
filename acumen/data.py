# The MIT License (MIT)
# © 2025 Acumen

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import random
import aiohttp
import asyncio
import urllib.parse
from async_lru import alru_cache
from typing import Any, Dict, List, Optional

MAX_ROWS_PER_REQUEST = 10

@alru_cache(maxsize=10000)
async def get_split_row_count(dataset, config="default", split="train"):
    """
    Uses the HuggingFace datasets-server API to retrieve the number of examples in a given dataset/config/split.
    """
    encoded_dataset = urllib.parse.quote(dataset)
    info_url = f"https://datasets-server.huggingface.co/info?dataset={encoded_dataset}"
    async with aiohttp.ClientSession() as session:
        async with session.get(info_url) as response:
            response.raise_for_status()
            data = await response.json()
    try:
        return data["dataset_info"][config]["splits"][split]["num_examples"]
    except KeyError:
        raise ValueError(f"Split '{split}' not found in dataset '{dataset}' with config '{config}'.")

async def fetch_batch(session, dataset, config, split, offset, length):
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    async with session.get(url, params=params) as response:
        response.raise_for_status()
        data = await response.json()
        return data["rows"]

async def get_samples(
    dataset="PrimeIntellect/verifiable-coding-problems", 
    config="default", 
    split="train", 
    sample_size=1,
    seed="42"
):
    """
    Fast parallel sampling from HuggingFace datasets-server.
    """
    total_rows = await get_split_row_count(dataset, config, split)
    sample_size = min(sample_size, total_rows-1)
    random.seed(seed)

    # Create randomized batch offsets
    num_batches = (sample_size + MAX_ROWS_PER_REQUEST - 1) // MAX_ROWS_PER_REQUEST
    offsets = sorted(random.sample(range(0, total_rows - MAX_ROWS_PER_REQUEST), num_batches))
    batch_lengths = [min(MAX_ROWS_PER_REQUEST, sample_size - i * MAX_ROWS_PER_REQUEST) for i in range(num_batches)]

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_batch(session, dataset, config, split, offset, length)
            for offset, length in zip(offsets, batch_lengths)
        ]
        results = await asyncio.gather(*tasks)

    # Flatten and truncate to exact sample_size
    all_samples = [item for batch in results for item in batch]
    return all_samples[:sample_size]


@alru_cache(maxsize=10_000)
async def get_split_row_count(
    dataset: str,
    config: str = "default",
    split: str = "train"
) -> int:
    encoded = urllib.parse.quote(dataset)
    url = f"https://datasets-server.huggingface.co/info?dataset={encoded}"
    async with aiohttp.ClientSession() as sess:
        async with sess.get(url) as resp:
            resp.raise_for_status()
            info = await resp.json()
    try:
        return info["dataset_info"][config]["splits"][split]["num_examples"]
    except KeyError:
        raise ValueError(f"No split {split} in {dataset}/{config}")

async def fetch_batch(
    session: aiohttp.ClientSession,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> List[Dict[str, Any]]:
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        payload = await resp.json()
        return payload["rows"]

class InfiniteAsyncDataLoader:
    """
    Infinite async loader: keeps buffer topped up with random samples forever.
    
    Usage:
        loader = InfiniteAsyncDataLoader(
            "AI-MO/NuminaMath-TIR",
            sample_batch_size=5,
            buffer_size=20
        )
        # in an async context:
        while True:
            sample = await loader.next()
            print(sample)
    """

    def __init__(
        self,
        dataset: str,
        config: str = "default",
        split: str = "train",
        sample_batch_size: int = 1,
        buffer_size: int = 10,
    ):
        self.dataset = dataset
        self.config = config
        self.split = split
        self.batch_size = sample_batch_size
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        self._session = aiohttp.ClientSession()
        self._producer_task = asyncio.get_event_loop().create_task(self._producer())

    async def _producer(self):
        """
        Forever:
          - pick a random offset
          - fetch a batch
          - put each row into the queue (awaiting if full)
        """
        total = await get_split_row_count(self.dataset, self.config, self.split)
        while True:
            offset = random.randrange(0, max(1, total - MAX_ROWS_PER_REQUEST))
            length = MAX_ROWS_PER_REQUEST
            try:
                rows = await fetch_batch(
                    self._session,
                    self.dataset,
                    self.config,
                    self.split,
                    offset,
                    length
                )
                for row in rows:
                    await self.queue.put(row)   # will block if buffer full
            except Exception as e:
                # you can log/handle backoff here
                print (e)
                await asyncio.sleep(1)

    async def __anext__(self) -> Dict[str, Any]:
        """
        Always returns the next available sample.
        Never raises StopAsyncIteration.
        """
        return await self.queue.get()

    def __aiter__(self):
        return self

    async def next(self) -> Dict[str, Any]:
        """
        Helper so you can write: sample = await loader.next()
        """
        return await self.__anext__()

    async def close(self):
        """
        Clean up the HTTP session and cancel producer.
        """
        self._producer_task.cancel()
        await self._session.close()