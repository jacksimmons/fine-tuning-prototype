# Helper functions to format our input dataset for fine-tuning
def create_prompt_formats(sample):
    # Format fields of the sample ("instruction", "output"), concatenate them
    # using two newlines.
    INTRO_BLURB = "Below is an instruction describing a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruct: Summarize the below conversation."
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"

    blurb = f"\n{INTRO_BLURB}"
    instruction = f"\n{INSTRUCTION_KEY}"
    input_context = f"{sample["dialogue"]}" if sample["dialogue"] else None
    response = f"{RESPONSE_KEY}\n{sample["summary"]}"
    end = f"{END_KEY}"

    parts = [blurb, instruction, input_context, response, end]
    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt

    return sample