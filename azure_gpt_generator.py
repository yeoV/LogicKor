import argparse
import pandas as pd
from typing import Any
import time
from openai import AsyncAzureOpenAI
import asyncio


parser = argparse.ArgumentParser()
parser.add_argument(
    "--template",
    help=" : Template File Location",
    default="./templates/template-EEVE.json",
)
parser.add_argument(
    "--model", help=" : Deployment Model name", default="gpt-4", type=str
)
parser.add_argument("--api_base", help=" : API Base", default="", type=str)
parser.add_argument("--api_key", help=" : API key", default="", type=str)
parser.add_argument("--api_version", help=" : API Version", default="", type=str)

args = parser.parse_args()


class GeneratorGPT:

    single_turn_template: str
    double_turn_template: str
    df_questions: pd.DataFrame
    client: Any

    def __init__(self) -> None:
        self.single_turn_template, self.double_turn_template = self._load_template()
        self.df_questions = self._load_questions()
        self.client = AsyncAzureOpenAI(
            api_key=args.api_key,
            api_version=args.api_version,
            azure_endpoint=args.api_base,
        )

    def _load_template(self):
        df_config = pd.read_json(args.template, typ="series")
        return df_config.iloc[0], df_config.iloc[1]

    def _load_questions(self) -> pd.DataFrame:
        return pd.read_json("questions.jsonl", lines=True)

    def format_single_turn_question(self, question):
        return self.single_turn_template.format(question[0])

    def format_double_turn_question(self, question, single_turn_output):
        return self.double_turn_template.format(
            question[0], single_turn_output, question[1]
        )

    async def inference(self, idx, question):

        print(f"{idx}/41 Inference Start")
        res = await self.client.chat.completions.create(
            model=args.model,
            temperature=0,
            top_p=0,
            messages=[{"role": "system", "content": question}],
        )
        print(f"{idx}/41 Inference Completed")

        return res.choices[0].message.content

    async def single_run(self):
        # single turn
        print("** Single-turn inference Start **")
        print("=" * 80)
        single_turn_questions = (
            self.df_questions["questions"].map(self.format_single_turn_question)
        ).to_list()

        # API 호출 제한으로 인한 10개로 갯수 제한
        results = []  # 최종 결과를 저장할 리스트
        batch_size = 10
        for i in range(0, len(single_turn_questions), batch_size):
            batch = single_turn_questions[i : i + batch_size]
            task_list = [
                asyncio.create_task(self.inference(idx, question))
                for idx, question in enumerate(batch, start=i)
            ]
            batch_results = await asyncio.gather(*task_list)
            results.extend(batch_results)  # 배치 결과를 최종 결과 리스트에 추가
            print(f"Batch {i//batch_size + 1} completed")

        return results

    # single run 이 완료된 이후에 수행되어야 함
    async def multi_run(self, single_turn_outputs):
        print("** Multi-turn inference Start **")
        print("=" * 80)
        multi_turn_questions = (
            self.df_questions[["questions", "id"]].apply(
                lambda x: self.format_double_turn_question(
                    x["questions"], single_turn_outputs[x["id"] - 1]
                ),
                axis=1,
            )
        ).to_list()
        results = []
        batch_size = 10
        for i in range(0, len(multi_turn_questions), batch_size):
            batch = multi_turn_questions[i : i + batch_size]
            task_lists = [
                asyncio.create_task(self.inference(idx, question=question))
                for idx, question in enumerate(batch, start=i)
            ]
            batch_results = await asyncio.gather(*task_lists)
            results.extend(batch_results)
        return results

    async def run(self):

        single_turn_outputs = await self.single_run()
        multi_turn_outputs = await self.multi_run(single_turn_outputs)
        # Create to json file
        df_output = pd.DataFrame(
            {
                "id": self.df_questions["id"],
                "category": self.df_questions["category"],
                "questions": self.df_questions["questions"],
                "outputs": list(zip(single_turn_outputs, multi_turn_outputs)),
                "references": self.df_questions["references"],
            }
        )
        df_output.to_json(
            f"{args.model}-{time.strftime('%y%m%d%H%M', time.localtime(time.time()))}.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )


async def main():
    s_time = time.time()
    generator_gpt = GeneratorGPT()
    await generator_gpt.run()

    print(f"All tasks are done.. processed time : {time.time() - s_time // 60} min")


if __name__ == "__main__":
    # 비동기 작업 시작
    asyncio.run(main())
