# Claude Code Project Instructions

## Core Behaviors
- **Web Search**: You MUST perform a web search for every user query to ensure you have the latest information. Do not rely solely on your training data. Verify facts via search before answering.
- **Auto-Approval**: The user has configured auto-approval for read/write/bash. Run commands proactively.

## Project Context
This is the "TextLLM" project, a terminal-based chatbot and FastAPI server using OpenAI's Realtime/Conversations Beta API.
- **Database**: PostgreSQL (Store metadata only)
- **API**: OpenAI `client.responses.create` (Stateful)
- **Server**: FastAPI (`server.py`)
- **Client**: Terminal (`terminal_chatbot.py`)

## Style
- Be concise.
- Fix bugs immediately.
