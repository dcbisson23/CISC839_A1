import config
import json
from transformers import AutoTokenizer

def get_data_from_response(response: str) -> dict:
    response_data = {}
    res_split = response
    # Get any thinking blocks first.
    res_split = res_split.split("</think>")
    thinking_blocks = res_split
    res_split = [thinking_blocks.pop(-1)]
    if len(thinking_blocks) > 0:
        for block in thinking_blocks:
            # Thinking blocks come before anything else, so this should work.
            block.removeprefix("<think>")
    response_data["reasoning_trace"] = "\n".join(thinking_blocks)
    res_split = res_split[-1].split("```")
    if len(res_split) > 1:
        # We're only going to evaluate on the last SQL segment if multiple exist.
        query = res_split[-2].removeprefix("sql")
    else:
        # If at this point we don't have any sql blocks, we'll assume the response is formatted correctly.
        query = res_split[-1]
    # We need the query to not have any newlines for the evaluation script to work on it.
    # Decided to also remove tabs because tabs spaces and newlines between keywords are treated the same in an SQL query.
    # Just makes it look nicer I think.
    query = query.replace("\n", " ")
    query = query.replace("\t", " ")
    query = " ".join(query.split())
    response_data["prediction"] = query.strip()

    return response_data







