import pytest
from gensh.processor import GenShell, load_api_tokens, search_duckduckgo
from unittest.mock import Mock, patch
import sqlite3
import os
import docx

@pytest.fixture
def mock_env_vars():
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_openai_key",
        "ANTHROPIC_API_KEY": "test_anthropic_key"
    }):
        yield

@pytest.fixture
def gensh(mock_env_vars):
    config = {
        'template_dir': 'tests/test_templates',
        'model': 'gpt-3.5-turbo',
        'execution_timeout': 5,
        'max_retries': 2,
        'retry_delay': 0.1,
        'db_path': ':memory:',  # Use in-memory SQLite database for testing
        'output_format': 'text'
    }
    return GenShell(config, verbose=True)

def test_load_api_tokens(mock_env_vars):
    tokens = load_api_tokens()
    assert tokens['OPENAI_API_KEY'] == 'test_openai_key'
    assert tokens['ANTHROPIC_API_KEY'] == 'test_anthropic_key'

@patch('gensh.processor.load_dotenv')
@patch('builtins.input', side_effect=['test_openai_key', 'test_anthropic_key'])
def test_load_api_tokens_from_input(mock_input, mock_load_dotenv):
    mock_load_dotenv.return_value = False
    with patch.dict(os.environ, {}, clear=True):
        tokens = load_api_tokens()
    assert tokens['OPENAI_API_KEY'] == 'test_openai_key'
    assert tokens['ANTHROPIC_API_KEY'] == 'test_anthropic_key'
    assert mock_input.call_count == 2

def test_gensh_initialization(gensh):
    assert isinstance(gensh, GenShell)
    assert gensh.verbose is True
    assert gensh.config['template_dir'] == 'tests/test_templates'
    assert isinstance(gensh.db_conn, sqlite3.Connection)
    assert gensh.api_tokens['OPENAI_API_KEY'] == 'test_openai_key'
    assert gensh.api_tokens['ANTHROPIC_API_KEY'] == 'test_anthropic_key'

@patch('gensh.processor.openai.OpenAI')
def test_generate_code_gpt(mock_openai, gensh):
    mock_openai.return_value.chat.completions.create.return_value.choices[0].message.content = "print('Hello, World!')"
    with patch('gensh.processor.search_duckduckgo', return_value=[]):
        code = gensh.generate_code("print hello world", "python")
    assert code == "print('Hello, World!')"
    assert len(gensh.conversation_history) == 2

@patch('gensh.processor.Anthropic')
def test_generate_code_claude(mock_anthropic, gensh):
    gensh.config['model'] = 'claude'
    mock_anthropic.return_value.messages.create.return_value.content[0].text = "print('Hello, Claude!')"
    with patch('gensh.processor.search_duckduckgo', return_value=[]):
        code = gensh.generate_code("print hello world", "python")
    assert code == "print('Hello, Claude!')"
    assert len(gensh.conversation_history) == 2

@patch('gensh.processor.ollama.chat')
def test_generate_code_ollama(mock_ollama, gensh):
    gensh.config['model'] = 'llama2'
    mock_ollama.return_value = {'message': {'content': "print('Hello, Ollama!')"}}
    with patch('gensh.processor.search_duckduckgo', return_value=[]):
        code = gensh.generate_code("print hello world", "python")
    assert code == "print('Hello, Ollama!')"
    assert len(gensh.conversation_history) == 2

def test_log_model_call(gensh):
    gensh.log_model_call("test_model", "test_query", "test_response")
    cursor = gensh.db_conn.cursor()
    cursor.execute("SELECT * FROM model_calls")
    result = cursor.fetchone()
    assert result is not None
    assert result[2] == gensh.session_id
    assert result[3] == "test_model"
    assert result[4] == "test_query"
    assert result[5] == "test_response"

def test_clear_history(gensh):
    gensh.conversation_history = [{"role": "user", "content": "test"}]
    gensh.do_clear_history("")
    assert len(gensh.conversation_history) == 0

def test_show_history(capsys, gensh):
    gensh.conversation_history = [
        {"role": "user", "content": "test query"},
        {"role": "assistant", "content": "test response"}
    ]
    gensh.do_show_history("")
    captured = capsys.readouterr()
    assert "User: test query" in captured.out
    assert "Assistant: test response" in captured.out

@patch('gensh.processor.subprocess.run')
def test_execute_python(mock_subprocess, gensh):
    mock_subprocess.return_value.stdout = "Hello, World!"
    mock_subprocess.return_value.stderr = ""
    output = gensh.execute_python("/tmp", "print('Hello, World!')")
    assert output == "Hello, World!"
    mock_subprocess.assert_called_once()

@patch('gensh.processor.subprocess.run')
def test_execute_shell(mock_subprocess, gensh):
    mock_subprocess.return_value.stdout = "Hello, World!"
    mock_subprocess.return_value.stderr = ""
    output = gensh.execute_shell("/tmp", "echo 'Hello, World!'")
    assert output == "Hello, World!"
    mock_subprocess.assert_called_once()

def test_process_command(gensh):
    gensh.process_piped_command = Mock()
    gensh.process_command('"print hello" | python')
    gensh.process_piped_command.assert_called_once_with("print hello", "python")

def test_set_output_format(gensh):
    gensh.do_set_output("markdown")
    assert gensh.output_format == "markdown"
    gensh.do_set_output("docx")
    assert gensh.output_format == "docx"
    gensh.do_set_output("invalid")
    assert gensh.output_format == "docx"  # Should not change for invalid input

def test_process_output_text(gensh):
    gensh.output_format = "text"
    input_md = "# Header\n\n```python\nprint('Hello')\n```\n\nSome text"
    expected_output = "Header\n\nSome text"
    assert gensh.process_output(input_md).strip() == expected_output

def test_process_output_markdown(gensh):
    gensh.output_format = "markdown"
    input_md = "# Header\n\n```python\nprint('Hello')\n```\n\nSome text"
    assert gensh.process_output(input_md) == input_md

@patch('gensh.processor.Document')
def test_process_output_docx(mock_document, gensh, tmp_path):
    gensh.output_format = "docx"
    input_md = "# Header\n\nSome text"
    
    mock_doc = Mock()
    mock_document.return_value = mock_doc
    
    result = gensh.process_output(input_md)
    
    mock_document.assert_called_once()
    mock_doc.add_paragraph.assert_called_once_with(input_md)
    mock_doc.save.assert_called_once()
    
    assert "Output saved to" in result

@patch('gensh.processor.requests.get')
def test_search_duckduckgo(mock_get):
    mock_get.return_value.text = '''
    <div class="result__body">
        <a class="result__a">Test Title</a>
        <a class="result__snippet">Test Snippet</a>
    </div>
    '''
    results = search_duckduckgo("test query")
    assert len(results) == 1
    assert results[0]['title'] == "Test Title"
    assert results[0]['snippet'] == "Test Snippet"

# Add more tests as needed
