import os
import openai
openai.api_base = "http://10.100.207.106:6010/" 
openai.api_key = os.environ["USER"] # not optional
prompts = [
    "De president van de Verenigde Staten is Donald",
    "Dit is een verhaal over"
]
output = openai.Completion.create(
  prompt=prompts,
  max_tokens=50,
  engine="opt"
)

for i, prompt in enumerate(prompts):
    print(f"Prompt: {prompt}")
    print(f"Completion: {output['choices'][i]['text']}")
