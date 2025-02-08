from asyncio import run

from core.train_test_set.collect_data import CollectData


async def main() -> None:
    return await CollectData().json_file()

if __name__ == '__main__':
    run(main())

# from transformers import pipeline
#
# model_name = 'tsmatz/xlm-roberta-ner-japanese'
# classifier = pipeline('token-classification', model=model_name)
# result = classifier('鈴井は4月の陽気の良い日に、鈴をつけて北海道のトムラウシへと登った')
# print(result)
