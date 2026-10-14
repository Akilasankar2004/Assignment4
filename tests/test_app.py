import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from app import MODEL, SYSTEM_PROMPT, build_chain, to_langchain_history


class CustomerSupportAppTests(unittest.TestCase):
    def test_history_preserves_user_and_assistant_turns(self):
        history = to_langchain_history(
            [
                {"role": "user", "content": "My order is late."},
                {"role": "assistant", "content": "Please share the order number."},
            ]
        )

        self.assertEqual(
            history,
            [
                HumanMessage(content="My order is late."),
                AIMessage(content="Please share the order number."),
            ],
        )

    def test_history_rejects_unsupported_roles(self):
        with self.assertRaises(ValueError):
            to_langchain_history([{"role": "system", "content": "unexpected"}])

    def test_chain_uses_gemini_and_includes_follow_up_history(self):
        observed = {}

        def fake_model(prompt_value):
            observed["messages"] = prompt_value.to_messages()
            return AIMessage(content="I can help with that.")

        fake_chat_model = RunnableLambda(fake_model)
        with patch("app.ChatGoogleGenerativeAI", return_value=fake_chat_model) as model:
            chain = build_chain("test-key")
            answer = chain.invoke(
                {
                    "history": [HumanMessage(content="My order is late.")],
                    "input": "Can I get a refund?",
                }
            )

        model.assert_called_once_with(
            model=MODEL,
            google_api_key="test-key",
            temperature=0,
        )
        self.assertEqual(answer, "I can help with that.")
        messages = observed["messages"]
        self.assertEqual(messages[0].content, SYSTEM_PROMPT)
        self.assertEqual(messages[1], HumanMessage(content="My order is late."))
        self.assertEqual(messages[2], HumanMessage(content="Can I get a refund?"))


if __name__ == "__main__":
    unittest.main()
