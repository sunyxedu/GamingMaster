"""API client for fetching replay metadata from generals.io."""

import json
import ssl
from typing import Any, Dict, List
from urllib.parse import quote

import aiohttp


async def fetch_replay_list(
    username: str,
    total: int = 10000,
    count_per_request: int = 200
) -> List[Dict[str, Any]]:
    """
    Fetch replay metadata list for a given username from generals.io API.
    
    Args:
        username: The player username
        total: Maximum number of replays to fetch
        count_per_request: Number of replays per API request
        
    Returns:
        List of replay metadata dictionaries
    """
    data = []
    num_requests = (total + count_per_request - 1) // count_per_request
    
    # Create SSL context that doesn't verify certificates (like axios default behavior)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(num_requests):
            offset = i * count_per_request
            # Last request: only fetch remaining count
            remaining = total - offset
            count = min(count_per_request, remaining)
            url = (f"https://generals.io/api/replaysForUsername"
                   f"?u={quote(username)}&offset={offset}&count={count}")
            
            try:
                async with session.get(url) as response:
                    replays = await response.json()
                    print(f"Request {i+1}: Retrieved {len(replays)} replays for {username}")
                    
                    if len(replays) == 0:
                        print('No more replays to retrieve. Stopping further requests.')
                        break
                        
                    data.extend(replays)
            except Exception as e:
                print(f"Error fetching replays at offset {offset}: {e}")
                continue
                
    return data


async def fetch_replay_pages(
    username: str,
    count_per_request: int = 200,
    start_offset: int = 0,
    stride_pages: int = 1,
):
    """Async generator yielding ``(next_offset, page)`` tuples, newest first,
    until the API is exhausted. Enables filter-as-you-go fetching.

    ``stride_pages > 1`` supports sharding: shard k of n starts at
    ``k * count_per_request`` with stride n, so shards page through disjoint
    slices of the player's history with zero coordination. ``next_offset``
    is what a resuming caller should pass back as ``start_offset``.
    """
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    connector = aiohttp.TCPConnector(ssl=ssl_context)
    # Bounded timeout so a hung/rate-limited page ends the run gracefully
    # instead of blocking the whole pipeline on an open socket forever.
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        offset = start_offset
        while True:
            url = (f"https://generals.io/api/replaysForUsername"
                   f"?u={quote(username)}&offset={offset}&count={count_per_request}")
            try:
                async with session.get(url) as response:
                    replays = await response.json()
            except Exception as e:
                print(f"Error fetching replays at offset {offset}: {e}")
                return
            if not replays:
                return
            offset += count_per_request * stride_pages
            yield offset, replays
            if len(replays) < count_per_request:
                return  # short page = history exhausted


async def save_replay_list(username: str, output_path: str, total: int = 10000):
    """
    Fetch and save replay list to a JSON file.
    
    Args:
        username: The player username
        output_path: Path to save the JSON file
        total: Maximum number of replays to fetch
    """
    replays = await fetch_replay_list(username, total)
    
    with open(output_path, 'w') as f:
        json.dump(replays, f, indent=4)
    
    print(f"Saved {len(replays)} replays for {username} to {output_path}")
    return replays

