import json
from tqdm import tqdm
from transformers import AutoTokenizer
from argparse import ArgumentParser
from pathlib import Path
import config


class SpiderPromptBuilder:
    def __init__(self, tables_path: str, questions_path: str):
        """
        Initializes the builder by loading the Spider dataset files.
        """
        with open(tables_path, 'r', encoding='utf-8') as f:
            self.tables_data = json.load(f)

        with open(questions_path, 'r', encoding='utf-8') as f:
            self.questions_data = json.load(f)

        # Create a lookup dictionary for quick schema retrieval by db_id
        self.schema_lookup = {db['db_id']: db for db in self.tables_data}

    def _format_schema_to_ddl(self, db_id: str) -> str:
        """
        Converts Spider's index-based schema into SQL CREATE TABLE statements.
        """
        db_info = self.schema_lookup[db_id]
        tables = db_info['table_names_original']
        columns = db_info['column_names_original']
        col_types = db_info['column_types']
        pks = set(db_info['primary_keys'])
        fks= {fk[0]: fk[1] for fk in db_info['foreign_keys']}

        # Map columns to their respective tables
        table_columns = {i: [] for i in range(len(tables))}
        for idx, (tbl_idx, col_name) in enumerate(columns):
            if tbl_idx == -1:  # Skip the '*' column
                continue

            # Map Spider's generic types to standard SQL types
            sql_type = {
                'text': 'TEXT',
                'number': 'NUMERIC',
                'time': 'TEXT',  # SQLite handles time as text usually
                'boolean': 'BOOLEAN',
                'others': 'BLOB'
            }.get(col_types[idx], 'TEXT')

            # The line in SQLite defining each column changes if it's a primary or foreign key.
            # Need to adjust accordingly.
            if idx in pks:
                table_columns[tbl_idx].append(f"  {col_name} {sql_type} PRIMARY KEY")
            elif idx in fks.keys():
                ref_idx = fks[idx]
                ref_table = tables[columns[ref_idx][0]]
                ref_col_name = columns[ref_idx][1]
                table_columns[tbl_idx].append(f"  FOREIGN KEY ({col_name}) REFERENCES {ref_table}({ref_col_name})")
            else:
                table_columns[tbl_idx].append(f"  {col_name} {sql_type}")

        # Build the CREATE TABLE statements
        ddl_statements = []
        for i, tbl_name in enumerate(tables):
            cols = ",\n".join(table_columns[i])
            ddl_statements.append(f"CREATE TABLE {tbl_name} (\n{cols}\n);")

        schema_str = f"""```sql\n{"\n".join(ddl_statements)}\n```"""

        return schema_str

    def build_chat_template(self, question: str, db_id: str, add_cot: bool = False) -> list[dict[str, str]]:
        """
        Constructs the chat template to be applied when prompting the LLM.
        """
        schema_str = self._format_schema_to_ddl(db_id)
        system_context = config.SYS_CONTEXT
        if add_cot:
            # The goal here is to give the model a CoT template to help guide it through answering the question.
            added_cot_context = config.COT_CONTEXT
            system_context += added_cot_context

        user_question = f"""DATABASE SCHEMA:\n{schema_str}\nQUESTION: \"{question}\"\n"""
        chat_template = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_question},
        ]
        return chat_template

    def generate_prompts(self, output_path: str,
                         tokenizer: AutoTokenizer,
                         sample_rate: float = 1.0,
                         add_cot: bool = False):
        """
        Iterates through dev.json and generates a JSONL file of prompts and ground truth.
        """
        with open(output_path, 'w', encoding='utf-8') as f_out:
            db_sample_accum = {}
            total_samples = 0
            for i, item in tqdm(enumerate(self.questions_data), desc="Generating prompts"):

                db_id = item['db_id']
                question = item['question']
                # We only want a # of samples for each db proportional to the # of questions available to the db.
                # So, take the first question for each db, then only take questions when enough questions have gone by.
                # Ex:
                # - for sample_rate = 1, take every question.
                # - for sample_rate = 0.5, take every other question.
                skip_flag = True
                if db_sample_accum.get(db_id) is None:
                    db_sample_accum[db_id] = 0.0
                    skip_flag = False
                else:
                    db_sample_accum[db_id] += sample_rate
                    if db_sample_accum[db_id] >= 1.0:
                        db_sample_accum[db_id] -= 1.0
                        skip_flag = False
                if skip_flag:
                    continue

                chat_template = self.build_chat_template(question, db_id, add_cot=add_cot)

                prompt = tokenizer.apply_chat_template(
                    chat_template,
                    tokenize=False,
                    add_generation_prompt=True
                )
                record = {
                    "id": item.get("question_id", i),
                    "db_id": db_id,
                    "prompt": prompt,
                }
                f_out.write(json.dumps(record) + "\n")
                total_samples += 1

        print(f"Successfully generated {total_samples} prompts to {output_path}")


# --- Example Usage ---
if __name__ == "__main__":
    import config
    from utils import load_tokenizer
    parser = ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        choices=[d.get("model") for d in config.MODELS] + ["all"],
                        default="all"
                        )
    parser.add_argument("--sample_rate",
                        type=float,
                        default=config.SAMPLE_RATE
                        )
    args = parser.parse_args()
    model_sel = str(args.model)
    for model_dict in config.MODELS:
        model_name = model_dict.get("model")

        if model_sel == model_name or model_sel == "all":
            output_dir = f"{config.DATA_ROOT}/{model_name}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = f"{output_dir}/{config.PROMPTS_FILENAME}"
            builder = SpiderPromptBuilder(config.TABLES_PATH, config.QUESTIONS_PATH)

            tokenizer = load_tokenizer(model_dict)

            builder.generate_prompts(output_path, tokenizer, args.sample_rate)
            # This prints the first prompt in the new jsonl, for verification.
            with open(output_path, 'r') as f:
                first_record = json.loads(f.readline())
                print("--- SAMPLE PROMPT ---")
                print(first_record['prompt'])

            if model_dict.get("can_reason"):
                cot_output_path = f"{output_dir}/{config.COT_PROMPTS_FILENAME}"
                builder.generate_prompts(cot_output_path, tokenizer, args.sample_rate, add_cot=True)

                with open(cot_output_path, 'r') as f:
                    first_record = json.loads(f.readline())
                    print("--- SAMPLE PROMPT ---")
                    print(first_record['prompt'])
