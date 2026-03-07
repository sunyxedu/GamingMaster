"""API client for fetching replay metadata from generals.io."""

import json
import ssl
from typing import Any, Dict, List

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
            url = f"https://generals.io/api/replaysForUsername?u={username}&offset={offset}&count={count}"
            
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

