from typing import Final

EMBEDDING_MODEL: Final[str] = 'BAAI/bge-small-en-v1.5'

RAG_DATA: Final[str] = 'data/rag_data.md'

SOURCE: Final[str] = 'https://www.asahi.com'
MAIN_LINK: Final[str] = (f'{SOURCE}/topics/AP-7274059d-8405-4d7f-8dbd-8203b01bbbc8/timeline-unit/735073dc-'
                         '92bf-485a-909c-0d823315cb69/?iref=com_topics_7274059d-8405-4d7f-8dbd-8203b01bbbc8_timeline_'
                         'readmore')
HEADERS: Final[dict[str, str]] = {
    'Accept': (
        'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,'
        'image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
    ),
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': (
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/131.0.0.0 Safari/537.36'
    )
}
