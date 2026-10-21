# YouTube Summariser

A LangChain-based project for summarising YouTube videos. This repository contains modular implementations of LangChain components including chat models, embeddings, LLMs, output parsers, prompts, runnables, structured output, and text splitters.

## Project Structure

```
youtube_summariser/
├── ChatModel/              # Chat model implementations
├── Embeddings/             # Embedding models
├── LLM/                    # Language model wrappers
├── Output_Parser/          # Output parsers (JSON, Pydantic, String)
├── Prompts/                # Prompt templates and chat prompts
├── Runnables/              # LangChain Runnable primitives
├── Structured_Output/      # Structured output examples
├── Text_Splitter/          # Text splitting strategies
├── test.py                 # Test script
├── requirements.txt        # Python dependencies
└── README.md
```

## Features

- 🤖 **Chat Models** – Integration with LangChain chat models
- 🔤 **Embeddings** – Text embedding generation
- 🧠 **LLMs** – Large language model interfaces
- 📝 **Output Parsers** – Parse LLM outputs into JSON, Pydantic, or string formats
- 💬 **Prompts** – Reusable prompt templates and chat prompt examples
- ⚡ **Runnables** – LangChain Runnable, Lambda, Parallel, Passthrough, and Branch
- 📊 **Structured Output** – Typed and structured LLM responses
- ✂️ **Text Splitter** – Length-based and structure-based text splitting

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

1. Clone the repository:

```bash
git clone https://github.com/akku1298/langchain.git
cd youtube_summariser
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your environment variables:

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
# or other provider keys depending on the model used
```

### Usage

Run the test script:

```bash
python test.py
```

Explore individual modules under their respective folders (e.g., `LLM/llm.py`, `Prompts/prompt_template.py`, `Runnables/parallel.py`).

## Requirements

See [`requirements.txt`](requirements.txt) for the full list of Python packages.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is open source and available under the MIT License.