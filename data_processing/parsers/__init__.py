"""Parsers for converting replay data to training format."""

from .generals_json import (
    convert_simulator_to_gio,
    parse_replay_to_gio,
    batch_parse_directory,
)

__all__ = ['convert_simulator_to_gio', 'parse_replay_to_gio', 'batch_parse_directory']