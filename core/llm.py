from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from agent import vector_search, calculate, read_file, run_python


llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

tools = [
    vector_search,
    calculate,
    read_file,
    run_python
]

agent = create_agent(
    model=llm,
    tools=tools
)

resp = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is 234 + 876?"}
    ]
})

print(resp)