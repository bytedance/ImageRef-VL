# %%
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import FewShotChatMessagePromptTemplate

# 假设你有一些示例对话
examples = [
    {"input": "Hello!", "output": "Hi! How can I assist you today?"},
    {"input": "What's the weather?", "output": "It's sunny today!"}
]

# 定义 example_prompt
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", '{{"role": "user", "content": "{input}"}}'),
        ("ai", '{{"role": "ai", "content": "{output}"}}'),
    ]
)

# 定义 FewShotChatMessagePromptTemplate
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 渲染 few_shot_prompt
formatted_prompt = few_shot_prompt.format_messages()

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# %%

from utils import convert_messages

print(convert_messages(final_prompt.invoke(input="hi").messages))

# %%