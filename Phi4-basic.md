# Basic setup for Phi4

## Hardware

These examples are run on a system with lowcost T4 GPU, which works fine for inference.

## Library Compatibility

The Phi4 library has specific compatibility requirements.
Based on guidance from https://huggingface.co/microsoft/Phi-4-mini-instruct#inference-with-transformers

```
torch==2.5.1
transformers==4.49.0
accelerate==1.3.0
```

And with additional information from https://github.com/pytorch/pytorch/wiki/PyTorch-Versions

```
torchvision==0.20.0
```

## Obtain Configuration

```
from transformers import AutoConfig
config = AutoConfig.from_pretrained("microsoft/Phi-4-mini-instruct")
print(config)
```

## Prompt

I have $20,000 in my savings account, where I receive a 4% profit per year and payments twice a year. Can you please tell me how long it will take for me to become a millionaire? Also, can you please explain the math step by step as if you were explaining it to an uneducated person?

## Basic Operation

```
pipeline = transformers.pipeline(
    "text-generation",
    model="microsoft/Phi-4-mini-instruct",
    model_kwargs={"torch_dtype": "auto"},
    device_map="auto",
)

messages_a = [
    {"role": "user", "content": "I have $20,000 in my savings account, where I receive a 4% profit per year and payments twice a year. Can you please tell me how long it will take for me to become a millionaire? Also, can you please explain the math step by step as if you were explaining it to an uneducated person?"},
]

outputs = pipeline(messages_a, max_new_tokens=2048)
print(outputs[0]["generated_text"][-1])
```

## Basic Output

Device set to use cuda:0

{'role': 'assistant', 'content': "Sure, I'd be happy to help you with that! Let's break it down step by step.\n\n1. You currently have $20,000 in your savings account.\n2. You receive a 4% profit per year, which means your money grows by 4% each year.\n3. You receive payments twice a year, so your money grows twice as fast as it would if you only received it once a year.\n\nTo calculate how long it will take for you to become a millionaire, we need to use the formula for compound interest:\n\nA = P(1 + r/n)^(nt)\n\nWhere:\nA = the final amount (in this case, $1,000,000)\nP = the initial amount (in this case, $20,000)\nr = the annual interest rate (in this case, 4% or 0.04)\nn = the number of times the interest is compounded per year (in this case, 2)\nt = the number of years\n\nWe want to solve for t, so let's rearrange the formula:\n\nt = (log(A/P)) / (n * log(1 + r/n))\n\nNow, let's plug in the values:\n\nt = (log(1,000,000/20,000)) / (2 * log(1 + 0.04/2))\n\nt = (log(50)) / (2 * log(1.02))\n\nt ≈ 79.43\n\nSo, it will take you approximately 79.43 years to become a millionaire with a 4% profit per year and payments twice a year.\n\nI hope this explanation helps! Let me know if you have any other questions."}

##

[Basic1-Phi4-mini-Setup-pub.ipynb)](https://github.com/pauldeadman/pd-ml-phi4-basic/blob/main/Basic1-Phi4-mini-Setup-pub.ipynb "Jupyter Notebook")

