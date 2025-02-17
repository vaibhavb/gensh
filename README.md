# GenSh - Generate and Execute Shell
GenSh is a command-line tool that allows you to generate and execute code using natural language prompts. It leverages large language models to translate your queries into Python or shell code and then executes the generated code. The intenation of this tool is to assist in software and security research.


## Requirements
GenSh requires the following:
- Python 3.11 or higher
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
gensh -c 'python_code "find your ip address" | exec_python'
```

### Prepare Templates for copying
To read inbuilt templates for processing by genai agents:
```
gensh -l book | pbcopy
```

### TODO: Batch Processing
To process commands from a file:
```
gensh -f example_batch.gensh
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

## Inspirations
- [Aider](https://aider.chat/)
- [Fabric](https://github.com/danielmiessler/fabric)
- [Security Prompts](https://learning.oreilly.com/library/view/chatgpt-for-cybersecurity/9781805124047/B21091_03.xhtml)

## BUGS / FEATURES / TODO
[x] Generate and execute Python or shell code from natural language queries
[x] Support for multiple LLM backends: OpenAI GPT, Anthropic Claude, and Ollama
[x] Web search integration to improve code generation
[x] Interactive shell mode
[x] Command-line execution mode
[x] Configurable settings via JSON configuration file
[x] Verbose mode for detailed output
[x] Error handling and retry logic for code generation
[x] Pre-defined templates for common tasks
[x] Logging of language model calls to SQLite database
[x] Session-based conversation history
[x] API token management using .env file
[x] Bug: Release base templates with the shell, list_templates is broken. DONE: added resources
[x] Bug: History functionality doesnt work - DONE: added show_model commands
[x] Feat: Ability to inject on variables in templates, ask for template input
[ ] Feat: Add Experimental MCP Support 
[ ] Feat: Add test cases for functionality like generate | parse_context | exec_code
[ ] Feat: Add an eval suite which is based on exercism.org code templates
[ ] Feat: Add voice capability
[ ] Feat: Multiple output formats: text, Markdown, and Word document (docx)
[ ] Feat: Batch processing of commands from a file
[ ] Feat: Refactor code
[ ] Feat: Update docs
[ ] Feat: Ability to edit files in the whole_format
[ ] Feat: Generate sensible message for git commit changes
[ ] Add analyze_logs use case to gensh
