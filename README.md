# AI Customer Support Assistant

A Streamlit chat app for e-commerce support queries. LangChain connects the
conversation to Google's Gemini API using the current supported model,
`gemini-3.8-flash`, and conversation history is retained for follow-up questions
within the current browser session. This starter does not connect to store
policies or order systems, so the assistant asks for context instead of claiming
access to orders or inventing specific policy outcomes.

## Setup on Windows

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Create and edit a local `.env` file:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   Set `GOOGLE_API_KEY` to your Gemini API key. The `.env` file is ignored by
   Git; do not commit or share it.

4. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

## Tests

```powershell
python -m unittest discover -s tests
```

## Try it

After configuring the key, try questions about a delayed order, a damaged
delivery, a return, and a refund. Check that answers ask for missing order
details, handle follow-up questions using the chat history, and do not invent
store-specific policies or promise refund outcomes.
