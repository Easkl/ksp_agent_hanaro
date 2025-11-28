import json
from pathlib import Path
from typing import Iterable


LOG_FILE = Path(__file__).with_suffix(".json")

def _load_history() -> list:
	if LOG_FILE.exists():
		try:
			data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
			if isinstance(data, list):
				return data
		except json.JSONDecodeError:
			pass
	return []


observation_history = _load_history()


def append_observation(observation: Iterable):
	observation_history.append(list(observation))
	_save_history()


def reset_history():
	observation_history.clear()
	_save_history()


def _save_history():
	LOG_FILE.write_text(
		json.dumps(observation_history, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)