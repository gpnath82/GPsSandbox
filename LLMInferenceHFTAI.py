from huggingface_hub import InferenceClient
 
client = InferenceClient(
    provider="together",
    api_key="xxxxxxx"
)
messages = [
    {
        "role": "user",
        "content": "Hello"


    }
]
completion = client.chat.completions.create(
    #model="deepseek-ai/DeepSeek-R1",
    model="deepseek-ai/DeepSeek-V3", 
    messages=messages,
    max_tokens=100
)
 
print(completion.choices[0].message)