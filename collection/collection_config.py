"""Load task CLI defaults; explicit command-line options retain precedence."""
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = Path(__file__).resolve().parent / 'configs'


def parse_task_args(parser, task, section):
    default = CONFIG_ROOT / f'{task}.json'
    parser.add_argument('--config', type=Path, default=default,
                        help='task JSON configuration; explicit CLI options override it')
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--config', type=Path, default=default)
    selected, _ = selector.parse_known_args()
    payload = json.loads(selected.config.read_text())
    if section not in payload or not isinstance(payload[section], dict):
        parser.error(f'{selected.config}: missing configuration section {section!r}')
    actions = {action.dest: action for action in parser._actions}
    defaults = {}
    for key, value in payload[section].items():
        if key not in actions or key in ('help', 'config'):
            parser.error(f'{selected.config}: unknown option {key!r}')
        action = actions[key]
        if action.choices is not None and value not in action.choices:
            parser.error(f'{selected.config}: invalid {key}={value!r}')
        if action.type is Path and value is not None:
            value = Path(value).expanduser()
            if not value.is_absolute():
                value = PROJECT_ROOT / value
        defaults[key] = value
    parser.set_defaults(**defaults)
    return parser.parse_args()
