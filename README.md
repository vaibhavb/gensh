# GenSh - Generate and Execute Shell
GenSh is a command-line tool that allows you to generate and execute code using natural language prompts. It leverages large language models to translate your queries into Python or shell code and then executes the generated code.

## Features
- Generate and execute Python or shell code from natural language queries
- Support for multiple LLM backends: OpenAI GPT, Anthropic Claude, and Ollama
- Web search integration to improve code generation
- Interactive shell mode
- Command-line execution mode
- Batch processing of commands from a file
- Configurable settings via JSON configuration file
- Verbose mode for detailed output
- Error handling and retry logic for code generation
- Pre-defined templates for common tasks
- Logging of language model calls to SQLite database
- Session-based conversation history
- API token management using .env file
- Multiple output formats: text, Markdown, and Word document (docx)

## TODO
- Ask for changes:
    - Add new features or test cases.
    - Describe a bug.
    - Paste in an error message or or GitHub issue URL.
    - Refactor code.
    - Update docs.
- Will edit your files to complete your request.
- Automatically git commits changes with a sensible commit messag    

## Requirements
GenSh requires the following:

- Python 3.7 or higher
- OpenAI API key (for GPT-3.5 or GPT-4 access)
- Anthropic API key (for Claude access)
- Ollama installed locally (for running local models)
- Dependencies (installed automatically):
  - openai>=1.0.0
  - anthropic>=0.3.0
  - PyYAML>=6.0
  - ollama>=0.1.0
  - python-dotenv>=0.19.0
  - markdown>=3.3.0
  - python-docx>=0.8.10
  - requests>=2.25.0
  - beautifulsoup4>=4.9.3

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gensh.git
   cd gensh
   ```

2. Install the package:
   ```
   pip install .
   ```

## Setting up API Keys
Create a `.env` file in the root directory of your project and add your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Usage

### Interactive Mode
To start the interactive shell:
```
gensh
```

### Command-Line Execution
To run a single command:
```
gensh -c '"find your ip address" | python'
```

### Batch Processing
To process commands from a file:
```
gensh -f example_batch.txt
```

### Output Formats
Specify the output format:
```
gensh --output markdown
gensh --output docx
```

### Verbose Mode
Enable verbose output:
```
gensh -v
```

### Custom Configuration
Use a custom configuration file:
```
gensh --config /path/to/custom_config.json
```

## Configuration
Create a configuration file at `~/.gensh_config.json` with the following structure:
```json
{
    "template_dir": "templates",
    "model": "gpt-3.5-turbo",
    "execution_timeout": 30,
    "max_retries": 3,
    "retry_delay": 1,
    "db_path": "gensh_logs.db",
    "output_format": "text"
}
```

## Development
To set up a development environment:
1. Clone the repository
2. Install dependencies: `pip install -e ".[dev]"` or `poetry install`
3. Run tests: `pytest tests/`

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## BUGS
-Release base templates with the shell, list_templates is broken.
[] history functionality doesnt work
