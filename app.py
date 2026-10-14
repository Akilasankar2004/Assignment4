import os

import streamlit as st
from dotenv import load_dotenv
from httpx import HTTPError
from langchain_core.chat_history import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAI,
    GoogleAPIError,
    GoogleAuthenticationError,
    GoogleGenerativeAIError,
    GoogleModelNotFoundError,
    GooglePermissionDeniedError,
    GoogleRateLimitError,
)


MODEL = "gemini-3.8-flash"
SYSTEM_PROMPT = """You are a helpful e-commerce customer support assistant.
Be empathetic, concise, and practical. Use the conversation history to understand
follow-up questions. Never invent order details, store policies, delivery dates,
or refund outcomes. If information is missing, ask a focused question or explain
how the customer can contact support. Do not ask for passwords or full payment
card details."""


def build_chain(api_key: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    model = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=api_key,
        temperature=0,
    )
    return prompt | model | StrOutputParser()


def to_langchain_history(messages: list[dict[str, str]]) -> list[BaseMessage]:
    history: list[BaseMessage] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
        else:
            raise ValueError(f"Unsupported conversation role: {role}")
    return history


def render_provider_error(error: Exception) -> None:
    if isinstance(error, (GoogleAuthenticationError, GooglePermissionDeniedError)):
        st.error("Gemini rejected the API key. Check GOOGLE_API_KEY in your .env file.")
    elif isinstance(error, GoogleRateLimitError):
        st.error(
            "Gemini is temporarily rate-limited or your quota is exhausted. "
            "Wait and retry, or check your Google AI billing/quota settings."
        )
    elif isinstance(error, GoogleModelNotFoundError):
        st.error(f"The configured Gemini model ({MODEL}) is unavailable.")
    elif isinstance(error, HTTPError):
        st.error("Could not connect to Gemini. Check your internet connection and try again.")
    elif isinstance(error, (GoogleGenerativeAIError, GoogleAPIError)):
        st.error(f"Gemini request failed: {error}")


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="AI Customer Support", page_icon="💬")
    st.title("AI Customer Support Assistant")
    st.caption("Ask about delayed orders, damaged products, returns, or refunds.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Set GOOGLE_API_KEY in your local .env file to connect to Gemini.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.sidebar.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    chain = build_chain(api_key)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("How can we help?")
    if question:
        previous_messages = st.session_state.messages.copy()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Preparing a response..."):
                try:
                    answer = chain.invoke(
                        {
                            "history": to_langchain_history(previous_messages),
                            "input": question,
                        }
                    )
                except (GoogleGenerativeAIError, GoogleAPIError, HTTPError, GoogleRateLimitError) as error:
                    render_provider_error(error)
                else:
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )


if __name__ == "__main__":
    main()
