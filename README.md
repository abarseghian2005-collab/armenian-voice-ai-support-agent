# Armenian Voice AI Support Agent

## Overview
This project is an Armenian voice AI assistant for bank support.

It understands Armenian speech and responds in Armenian, but is strictly limited to:
- վարկեր (credits)
- ավանդներ (deposits)
- մասնաճյուղեր (branches)

The assistant does not answer anything outside these topics.

---

## How It Works

Pipeline:

User (voice)
→ Speech-to-Text (Whisper)
→ `ask()` logic (rag.py)
→ Retrieval from local data
→ LLM (only with that context)
→ Text-to-Speech

The voice agent itself does not generate answers — it only forwards input to the `ask()` function.

---

## Architecture

### `agent.py`
- Built with LiveKit
- Handles voice interaction
- Uses Whisper for Armenian STT
- Uses OpenAI TTS for output
- Calls `ask()` for all responses
- Starts with a greeting

---

### `rag.py`
Main logic of the system.

It:
- normalizes Armenian input (handles mistakes like "բանգ")
- detects bank and topic
- retrieves relevant text chunks
- sends only those chunks to the model
- formats the answer for TTS

---

### Data & Retrieval
- Uses ChromaDB for vector search
- Data is stored in `/data` as `.txt` files
- Each file corresponds to a bank + topic

---

## Data Collection (Important Note)

I initially tried to collect data automatically using:
- requests + BeautifulSoup
- Selenium (for dynamic pages)

However, many bank websites:
- had content behind clicks
- loaded data dynamically
- hid important sections in UI elements

Because of this, scraping was not reliable and often returned incomplete data.

To ensure the assistant gives correct and grounded answers, I manually collected the information from official bank websites and saved it into structured `.txt` files.

These files are included in the repository and are used as the knowledge base.

Apologies for not fully automating this part — I prioritized accuracy and completeness of the data.

---

## Guardrails

The assistant:
- only answers about loans, deposits, and branches
- refuses unrelated questions
- uses only provided data
- does not rely on external knowledge

---

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
## Setup

1. Install dependencies:
pip install -r requirements.txt

2. Create `.env` file and add your API keys

3. Load data:
python reload_db.py

4. Run the agent:
python agent.py
