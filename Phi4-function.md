# Phi4 Call Agents/Functions

The library versions are the same as per Basic example.

```
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig,pipeline,AutoTokenizer

model_path = "microsoft/Phi-4-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    attn_implementation="eager",
    torch_dtype="auto",
    trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
```

## Basic Data Setup

```
# Tools should be a list of functions stored in json format
tools = [
    {
        "name": "get_match_result",
        "description": "get match result",
        "parameters": {
            "match": {
                "description": "The name of the match",
                "type": "str",
                "default": "Arsenal vs ManCity"
            }
        }
    },
]

# Function implementations

def get_match_result(match: str) -> str:
    # This would be replaced by a weather API
    match_data = {
        "Arsenal vs ManCity": "1:1",
        "Chelsea vs ManUnited": "0:2"
    }
    return match_data.get(match, "I don't know")


messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant",
        "tools": json.dumps(tools), # pass the tools into system message using tools argument
    },
    {
        "role": "user",
        "content": "What is the result of Arsenal vs ManCity today?"
    }
]
```

## Convert Prompts

list of role:, content: ... prompts into native chat prompts for ROLES: user [from user], assistant [from model], system [how model should act - prefix], tool_call [requests for help]

```
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
```

## Add Device information

Iterate over the tokenizer output and add the hardware details. model.device will supply the information as per the model design.

```
model.device
```

## Call Inference

```
model.generate(**inputs, max_new_tokens=128)
```

## Decode Reponse

```
tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
```

## Extract Response

```
json.loads(response)[0]
```

## Call Function Identified

```
function_name = tool_call["name"]
arguments = tool_call["arguments"]
result = get_match_result(**arguments)
```



# Real Example

[Basic1-Phi4-mini-functon-debug-pub.ipynb](https://github.com/pauldeadman/pd-ml-phi4-basic/blob/main/Basic1-Phi4-mini-functon-debug-pub.ipynb)
