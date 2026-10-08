from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
import json
from tqdm import tqdm
from scripts.utils import load_model, load_tokenizer


class LLMPrompter:
    def __init__(self, model:AutoModelForCausalLM, tokenizer:AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = {}

    def load_gen_config(self, gen_config_path:str):
        with open(gen_config_path, 'r') as f:
            self.gen_config = json.load(f)

    def prompt(self, prompt:str) -> str:
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            **self.gen_config
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response

    def run_prompt_pipeline(self, prompts_path:str, output_path:str):
        with open(prompts_path, 'r') as f:
            prompts = f.readlines()
        with open(output_path, "w+") as f_out:
            # We want a list of existing responses to make the pipeline hotstart for incomplete response files.
            existing_outputs = f_out.readlines()
            existing_ids = []
            for output_json in existing_outputs:
                existing_output_record = json.loads(output_json)
                existing_ids.append(existing_output_record["id"])
            for prompt_json in tqdm(prompts, desc=f"Generating responses"):
                prompt_record = json.loads(prompt_json)
                id = prompt_record["id"]
                db_id = prompt_record["db_id"]
                prompt = prompt_record["prompt"]
                # We don't need to redo any existing responses.
                if id in existing_ids:
                    existing_ids.remove(id)
                    continue

                response = self.prompt(prompt)
                output_record = {
                    "id": id,
                    "db_id": db_id,
                    "response": response,
                }
                f_out.write(json.dumps(output_record) + "\n")
        print(f"Successfully wrote {len(prompts)} responses to {output_path}")


if __name__ == "__main__":
    import config
    parser = ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        choices=[d.get("model") for d in config.MODELS] + ["all"] + ["all"],
                        default="all"
                        )
    parser.add_argument("--data_root",
                        type=str,
                        default=config.DATA_ROOT)
    parser.add_argument("--prompts_filename",
                        type=str,
                        default=config.PROMPTS_FILENAME)
    parser.add_argument("--responses_filename",
                        type=str,
                        default=config.RESPONSES_FILENAME)
    parser.add_argument("--gen_config_filename",
                        type=str,
                        default=config.GEN_CONFIG_FILENAME)

    args = parser.parse_args()
    model_sel = str(args.model)
    data_root = str(args.data_root)
    prompts_filename = str(args.prompts_filename)
    responses_filename = str(args.responses_filename)
    config_filename = str(args.gen_config)
    for model_dict in config.MODELS:
        if model_sel == model_dict['model'] or model_sel == "all":
            gen_config_path = f"{data_root}/{model_dict['model']}/{config_filename}"
            prompts_path = f"{data_root}/{model_dict['model']}/{prompts_filename}"
            responses_path = f"{data_root}/{model_dict['model']}/{responses_filename}"
            print("Loading model "+model_dict['model'])
            model = load_model(model_dict)
            tokenizer = load_tokenizer(model_dict)
            prompter = LLMPrompter(model, tokenizer)
            prompter.load_gen_config(gen_config_path)
            prompter.run_prompt_pipeline(prompts_path, responses_path)

            if model_dict.get("can_reason"):
                cot_prompts_path = f"{config.DATA_ROOT}/{model_dict['model']}/{config.COT_PROMPTS_FILENAME}"
                cot_responses_path = f"{config.DATA_ROOT}/{model_dict['model']}/{config.COT_RESPONSES_FILENAME}"
                prompter.run_prompt_pipeline(cot_prompts_path, cot_responses_path)
