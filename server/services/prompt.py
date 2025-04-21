from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

Socrates_prompt = PromptTemplate(
    input_variables=["input"],
    template="You are Socrates. {input}",
)
Socrates_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are Socrates."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="history"),
    ]
)
