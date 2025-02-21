from typing import Final

FILE_1_PATH: Final[str] = 'core/data/processed_data.json'

JAPANESE_SEPARATOR: Final[str] = 'cl-tohoku/bert-base-japanese'

JAPANESE_PUNCTUATION: Final[list[str]] = ['「', '」', '、',  '。', '！', '!', '?', '? ', '.', '_']
REGEX_NUMBERS: Final[str] = r'\d+'
