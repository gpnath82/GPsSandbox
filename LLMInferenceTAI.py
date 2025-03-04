from together import Together

client = Together()

response = client.chat.completions.create(
    model="deepseek-ai/deepseek-llm-67b-chat",
    messages=[],
    max_tokens=512,
    temperature=0.7,
    top_p=0.7,
    top_k=50,
    repetition_penalty=1,
    stop=["<｜begin▁of▁sentence｜>","<｜end▁of▁sentence｜>"],
    stream=True
)
for token in response:
    if hasattr(token, 'choices'):
        print(token.choices[0].delta.content, end='', flush=True)