from openai import OpenAI

client = OpenAI(api_key="sk-xxx", base_url="http://18.191.125.135:3000/v1/")

messages = [
    {"role": "system", "content": "你是一个乐于助人的体育界专家。"},
    {"role": "user", "content": "2008年奥运会是在哪里举行的？"},
]

data = client.chat.completions.create(
    model="deepseek-r1",
    stream=False,
    messages=messages
)

print(data.choices[0].message)