"""Replay fetcher for downloading and deserializing .gior files from S3."""

import asyncio
import aiohttp
import ssl
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from .deserializer import deserialize_replay


class ReplayFetcher:
    """Handles downloading and deserializing replay files."""
    
    def __init__(
        self,
        server: str = 'na',
        max_retries: int = 3,
        retry_delay: float = 2.0,
        max_concurrent: int = 10
    ):
        """
        Initialize the replay fetcher.
        
        Args:
            server: Server location ('na', 'eu', etc.)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            max_concurrent: Maximum concurrent downloads
        """
        self.base_url = f"https://generalsio-replays-{server}.s3.amazonaws.com"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create SSL context (like axios behavior)
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
    async def fetch_replay(
        self,
        session: aiohttp.ClientSession,
        replay_id: str,
        retry_count: int = 0
    ) -> Optional[dict]:
        """
        Fetch and deserialize a single replay.
        
        Args:
            session: aiohttp session
            replay_id: The replay ID
            retry_count: Current retry attempt
            
        Returns:
            Deserialized replay data or None if failed
        """
        url = f"{self.base_url}/{replay_id}.gior"
        
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.read()
                    return deserialize_replay(data)
                else:
                    raise Exception(f"HTTP {response.status}")
        except Exception as e:
            if retry_count < self.max_retries:
                await asyncio.sleep(self.retry_delay * (retry_count + 1))
                return await self.fetch_replay(session, replay_id, retry_count + 1)
            else:
                print(f"Failed to fetch {replay_id} after {self.max_retries} attempts: {e}")
                return None
    
    async def download_replay(
        self,
        session: aiohttp.ClientSession,
        replay_id: str,
        output_dir: Path
    ) -> bool:
        """
        Download and save a single replay with concurrency control.
        
        Args:
            session: aiohttp session
            replay_id: The replay ID
            output_dir: Directory to save the replay
            
        Returns:
            True if successful, False otherwise
        """
        async with self.semaphore:
            output_file = output_dir / f"{replay_id}.json"
            
            # Skip if already exists
            if output_file.exists():
                return True
            
            replay_data = await self.fetch_replay(session, replay_id)
            
            if replay_data:
                with open(output_file, 'w') as f:
                    json.dump(replay_data, f, indent=4)
                print(f"Successfully downloaded: {replay_id}")
                return True
            else:
                return False
    
    async def download_replays(
        self,
        replay_ids: List[str],
        output_dir: str,
        failed_log: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Download multiple replays concurrently.
        
        Args:
            replay_ids: List of replay IDs to download
            output_dir: Directory to save replays
            failed_log: Optional path to log failed downloads
            
        Returns:
            Dictionary with 'success' and 'failed' counts
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        failed_ids = []
        success_count = 0
        
        connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self.download_replay(session, replay_id, output_path)
                for replay_id in replay_ids
            ]
            
            results = await asyncio.gather(*tasks)
            
            for replay_id, success in zip(replay_ids, results):
                if success:
                    success_count += 1
                else:
                    failed_ids.append(replay_id)
        
        # Log failed downloads if requested
        if failed_log and failed_ids:
            with open(failed_log, 'w') as f:
                json.dump(failed_ids, f, indent=2)
            print(f"Logged {len(failed_ids)} failed downloads to {failed_log}")
        
        return {
            'success': success_count,
            'failed': len(failed_ids)
        }


async def download_replays(
    replay_ids: List[str],
    output_dir: str,
    server: str = 'na',
    max_concurrent: int = 10
) -> Dict[str, int]:
    """
    Convenience function to download replays.
    
    Args:
        replay_ids: List of replay IDs to download
        output_dir: Directory to save replays
        server: Server location
        max_concurrent: Maximum concurrent downloads
        
    Returns:
        Dictionary with 'success' and 'failed' counts
    """
    fetcher = ReplayFetcher(server=server, max_concurrent=max_concurrent)
    return await fetcher.download_replays(replay_ids, output_dir)

