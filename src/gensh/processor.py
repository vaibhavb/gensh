import yaml
import re
import os
import subprocess
from typing import List, Dict, Any
import openai
import cmd
import readline
import tempfile
import json
import time
from pathlib import Path
import sqlite3
import uuid
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import urllib.parse
import sys
import io
import importlib.metadata

def load_pattern_files(pattern_dir: str, pattern_name: str) -> Dict[str, str]:
    """Load system.md, user.md, and README.md files for a pattern.
    Args:
        pattern_dir: Base directory for patterns (e.g. ~/.gensh/fabric/patterns/)
        pattern_name: Name of the pattern folder (e.g. 'ai')
    Returns:
        Dictionary containing 'system', 'user', and 'readme' content
    """
    pattern_path = Path(os.path.expanduser(pattern_dir)) / pattern_name
    if not pattern_path.exists():
        print(f"Pattern directory '{pattern_name}' not found in {pattern_dir}")
        return {}
    files = {
        'system': pattern_path / 'system.md',
        'user': pattern_path / 'user.md',
        'readme': pattern_path / 'README.md'
    }
    content = {}
    for key, file_path in files.items():
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    content[key] = f.read().strip()
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            if (key != 'readme') and (key != 'user'):  # README and USER is optional
                print(f"Warning: {file_path} not found")
            content[key] = ""
    return content

def load_api_tokens():
    load_dotenv()
    tokens = {}
    for key in ['OPENAI_API_KEY']:
        tokens[key] = os.getenv(key)
        if tokens[key] is None:
            tokens[key] = input(f"Please enter your {key}: ").strip()
            if not tokens[key]:
                print(f"Warning: {key} not provided. Some functionality may be limited.")
    return tokens

def requires_confirmation(method):
    """Decorator to mark commands that require user confirmation."""
    method.requires_confirmation = True
    return method

def alias(*aliases):
    """Decorator to define aliases for commands."""
    def decorator(func):
        func.aliases = aliases
        return func
    return decorator

class GenShell(cmd.Cmd):
    prompt = "gensh> "
    fence = "---"
    defaultConfig = {
        "template_dir": "templates",
        "model": "gpt-3.5-turbo",
        "execution_timeout": 30,
        "max_retries": 3,
        "retry_delay": 1,
        "db_path": "~/.config/gensh/gensh_logs.db",
        "output_format": "text",
        "patterns_dir": "~/.config/gensh/fabric/patterns/"
    }

    def __init__(self, version: str, config: Dict[str, Any], verbose: bool = False):
        super().__init__()
        self.version = version
        self.config = config
        if not self.config:
            self.config = self.defaultConfig
        self.verbose = verbose
        self.api_tokens = load_api_tokens()
        self.openai_client = openai.OpenAI(api_key=self.api_tokens['OPENAI_API_KEY'])
        self.context = None
        self.session_id = str(uuid.uuid4())
        self.db_conn = self.init_database()
        self.templates = self.load_templates()
        self.history = []
        self.intro = f"Welcome to GenSh {self.version}. Type help or ? to list commands.\n"

    def init_database(self):
        db_path = self.config.get('db_path')
        db_path = os.path.expanduser(db_path)
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                session_id TEXT,
                model TEXT,
                query TEXT,
                response TEXT
            )
            ''')
            conn.commit()
            return conn
        except sqlite3.Error as e:
            print(f"Error initializing database: {e}")

    def log_model_call(self, model: str, query: str, response: str):
        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT INTO model_calls (timestamp, session_id, model, query, response)
            VALUES (datetime('now'), ?, ?, ?, ?)
        ''', (self.session_id, model, query, response))
        self.db_conn.commit()

    def do_show_model_response(self, query: str) -> str :
        "Show the last [num] model responses, default is 1, saves to context"
        num = 1
        if query != "":
            num = int(query)
        cursor = self.db_conn.cursor()
        cursor.execute('SELECT response FROM model_calls ORDER BY id DESC LIMIT ?', (num,))
        rows = cursor.fetchall()
        if rows:
            ctx = ""
            column_names = [description[0] for description in cursor.description]
            for row in rows:
                for col_name, value in zip(column_names, row):
                    ctx += value
                    print(f"{value}")
                print("-" * 40)  # Divider between rows for clarity
            self.context = ctx
        else:
            print("No records found.")

    def do_show_model_call(self, query: str) -> str:
        "Show the last [num] model calls, default is 1"
        num = 1
        if query != "":
            num = int(query)
        cursor = self.db_conn.cursor()
        cursor.execute('SELECT * FROM model_calls ORDER BY id DESC LIMIT ?', (num,))
        rows = cursor.fetchall()
        if rows:
            column_names = [description[0] for description in cursor.description]
            for row in rows:
                for col_name, value in zip(column_names, row):
                    print(f"{col_name}: {value}")
                print("-" * 40)  # Divider between rows for clarity
        else:
            print("No records found.")

    def do_context_parse(self, query):
        """Parse a file, extract py, sh, md or js code between triple backticks, and save them as appropriate file types. Saves to context."""
        file_content = self.context
        if (file_content == None or ""): return "No content to parse"
        # Regular expression to find code blocks in python, js, or markdown within triple backticks
        # TODO: FIX fencetag should not be hardcoded!
        code_blocks = re.findall(r'```(python|js|sh|md)\n(.*?)```', file_content, re.DOTALL)
        if code_blocks:
            code_file = ""
            for i, (language, code) in enumerate(code_blocks):
                # Set filetype based on the captured language
                filetype = {
                    'python': 'py',
                    'js': 'js',
                    'md': 'md',
                    'sh': 'sh'
                }.get(language, None)

                if filetype:
                    # Define the filename where each code block will be saved
                    filename = f"extracted_code_{i+1}.{filetype}"
                    code_file += (code.strip() + '\n')
                    print(f"Code block {i+1} ({language}) saved in {filename}:\n{code_file}")
                else:
                    print(f"Unrecognized language: {language}")
            #TODO: Fix this when there are multiple code blocks in model output
            self.context = code_file
        else:
            print("No code blocks found.")

    def replace_context_variables_in_query(self, query: str):
        # Find all placeholders in the form of $key using regex
        placeholders = re.findall(r'\$(\w+)', query) 
        # Replace each placeholder with its corresponding value in self.context
        for placeholder in placeholders:
            if placeholder in self.context:
                # Replace the $placeholder with the value from self.context
                query = query.replace(f"${placeholder}", str(self.context[placeholder]))
            else:
                print(f"Warning: '{placeholder}' not found in context.")
        
        return query

    def template_match(query: str):
        template_match = re.match(r'use template (\w+)(.*)', query, re.IGNORECASE)
        if template_match:
            template_name = template_match.group(1)
            template_args = template_match.group(2).strip()
            if template_name in self.templates:
                template = self.templates[template_name]
                prompt = template['prompt']
                #TODO: Don't ask user for the variables if they are already in context.
                for var, desc in template['variables'].items():
                    value = input(f"Enter {desc}: ")
                    prompt = prompt.replace(f"{{{var}}}", value)
                system_prompt = f"You are a {execution_type} code generator. Use the following template to generate code:\n\n{prompt}"
                user_prompt = f"*** Generate {execution_type} based on the template ***. {template_args}"
            else:
                print(f"Template '{template_name}' not found.")
                return None
        else:
            query = self.replace_context_variables_in_query(query)

    @alias("code")
    def do_python_code(self, query) -> str:
        "Python code generator"
        self.generate_code(query, 'python')
    
    @alias("bash")
    def do_shell_code(self, query) -> str:
        "Shell code generator"
        self.generate_code(query, 'shell')

    def generate_code(self, query: str, execution_type: str = 'python') -> str:
        "Python code generator."
        system_prompt = ""
        code_type = f"code-{execution_type}"
        prompt = self.get_prompt_template(code_type)['user_prompt']
        user_prompt = prompt + query;
        model = self.config.get('model', 'gpt-3.5-turbo')
        max_retries = self.config.get('max_retries', 3)
        retry_delay = self.config.get('retry_delay', 1)
        for attempt in range(max_retries):
            try:
                if self.verbose:
                    print(f"Generating code using {model} (attempt {attempt + 1}/{max_retries})...")
                if model.startswith('gpt'):
                    response = self.openai_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    code = response.choices[0].message.content.strip()
                elif model == 'claude':
                    response = self.anthropic_client.messages.create(
                        model="claude-3-sonnet-20240229",
                        max_tokens=1000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                    code = response.content[0].text.strip()
                else:  # Assume it's an Ollama model
                    response = ollama.chat(model=model, messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ])
                    code = response['message']['content'].strip()
                self.log_model_call(model, query, code)
                print(code)
                break
            except Exception as e:
                if self.verbose:
                    print(f"Error generating code: {e}")
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to generate code after {max_retries} attempts.")

    def do_use_template(self, arg):
        """Use a template by its alias or name. Usage: use-template alias_or_name [additional_args]"""
        if not arg:
            print("Usage: use-template alias_or_name [additional_args]")
            return
            
        parts = arg.split(maxsplit=1)
        alias_or_name = parts[0]
        additional_args = parts[1] if len(parts) > 1 else ""
        
        template_name = None
        if template_name is None:
            if alias_or_name in self.templates:
                template_name = alias_or_name
            else:
                print(f"No template or alias found for '{alias_or_name}'")
                return
        
        # Use the template
        template = self.templates[template_name]
        prompt = template['system_prompt']
        
        # Handle template variables
        for var, desc in template.get('variables', {}).items():
            if (self.context is not None) and (var in getattr(self, 'context', {})):
                value = self.context[var]
            else:
                value = input(f"Enter {desc}: ")
            prompt = prompt.replace(f"{{{var}}}", value)
        
        # Prepare and execute the model query
        system_prompt = f""
        user_prompt = f"{prompt}\n{additional_args}"
        
        model = self.config.get('model', 'gpt-3.5-turbo')
        max_retries = self.config.get('max_retries', 3)
        retry_delay = self.config.get('retry_delay', 1)
        print(system_prompt, user_prompt)
        self._do_model_call(model, max_retries, retry_delay, system_prompt, user_prompt, template_name)

    def do_list_patterns(self, arg):
        """List all available patterns in the patterns directory."""
        patterns_dir = os.path.expanduser(self.config.get('patterns_dir'))
        pattern_path = Path(patterns_dir)
        
        if not pattern_path.exists():
            print(f"Patterns directory not found: {patterns_dir}")
            return
            
        print("Available patterns:")
        for item in pattern_path.iterdir():
            if item.is_dir() and (item / 'system.md').exists():
                pattern_name = item.name
                readme_path = item / 'README.md'
                description = ""
                if readme_path.exists():
                    with open(readme_path, 'r') as f:
                        description = f.readline().strip()  # Get first line of README
                print(f"- {pattern_name}: {description}")

    def do_show_pattern(self, arg):
        """Show the content of a specific pattern's files."""
        if not arg:
            print("Error: must provide a pattern name")
            return
            
        patterns_dir = os.path.expanduser(self.config.get('patterns_dir'))
        pattern_content = load_pattern_files(patterns_dir, arg)
        
        if pattern_content:
            if pattern_content.get('readme'):
                print("README:")
                print("-" * 40)
                print(pattern_content['readme'])
                print()
                
            print("System Prompt:")
            print("-" * 40)
            print(pattern_content.get('system', 'Not found'))
            print()
            
            print("User Template:")
            print("-" * 40)
            print(pattern_content.get('user', 'Not found'))
        else:
            print(f"Pattern '{arg}' not found")

    @alias("gen")
    def do_exec_pattern(self, arg):
        """Execute a pattern with prompt on an AI model. Usage: exec-pattern pattern_name 'prompt'"""
        if not arg:
            print("Usage: exec-pattern pattern_name 'prompt'")
            return

        parts = arg.split(maxsplit=1)
        pattern_name = parts[0]
        prompt = parts[1] if len(parts) > 1 else ""
        
        # Check for piped input
        if not sys.stdin.isatty():
            # Read from pipe and add to args
            piped_input = sys.stdin.read().strip()
            prompt += piped_input

        patterns_dir = os.path.expanduser(self.config.get('patterns_dir'))
        pattern_content = load_pattern_files(patterns_dir, pattern_name)

        if not pattern_content:
            print(f"Pattern '{pattern_name}' not found")
            return

        system_prompt = pattern_content.get('system', "")
        user_prompt = pattern_content.get('user', "") + prompt 

        model = self.config.get('model', 'gpt-3.5-turbo')
        max_retries = self.config.get('max_retries', 3)
        retry_delay = self.config.get('retry_delay', 1)
        self._do_model_call(model, max_retries, retry_delay, system_prompt, user_prompt, pattern_name)


    def _do_model_call(self, model, max_retries, retry_delay, system_prompt,user_prompt, template_name):
        for attempt in range(max_retries):
            try:
                if self.verbose:
                    print(f"Generating code using {model} (attempt {attempt + 1}/{max_retries})...")

                if model.startswith('gpt'):
                    response = self.openai_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    code = response.choices[0].message.content.strip()
                elif model == 'claude':
                    response = self.anthropic_client.messages.create(
                        model="claude-3-sonnet-20240229",
                        max_tokens=1000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                    code = response.content[0].text.strip()
                else:  # Assume it's an Ollama model
                    response = ollama.chat(model=model, messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ])
                    code = response['message']['content'].strip()

                self.log_model_call(model, user_prompt, code)
                
                print(f"Generated using - '{template_name}':")
                print(code)
                break
                
            except Exception as e:
                if self.verbose:
                    print(f"Error generating code: {e}")
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Error generating code: {e}")
                    print(f"Failed to generate code after {max_retries} attempts.")

    
    def strip_markdown_code_block(self, code: str) -> str:
        code = re.sub(r'^```[\w]*\n', '', code)
        code = re.sub(r'\n```$', '', code)
        return code.strip()

    def confirm_execution(self) -> bool:
        from sys import stdout, stdin
        prompt = "Do you want to execute this code? (yes/no): "
        while True:
            stdout.write(prompt)
            stdout.flush()
            response = stdin.readline().strip().lower()
            if response in {'yes', 'y'}:
                return True
            elif response in {'no', 'n'}:
                return False
            stdout.write("Please answer 'yes' or 'no'.\n")
    
    def execute_code(self, code: str, execution_type: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            if execution_type == "python":
                self.execute_python(tmpdir, code)
            elif execution_type == "shell":
                self.execute_shell(tmpdir, code)

    def execute_python(self, tmpdir: str, code: str):
        main_file = os.path.join(tmpdir, "main.py")
        with open(main_file, "w") as f:
            f.write("#!/usr/bin/env python3\n")
            f.write(code)
        os.chmod(main_file, 0o755)  # Make the file executable
        try:
            timeout = self.config.get('execution_timeout', 30)
            if self.verbose:
                print(f"Executing Python code with timeout of {timeout} seconds...")
            result = subprocess.run([main_file], capture_output=True, text=True, timeout=timeout)
            print(result.stdout + result.stderr)
            return
        except subprocess.TimeoutExpired:
            print("Execution timed out after {timeout} seconds.")
            return
        except Exception as e:
            print(f"An error occurred during execution: {e}")
            return

    def execute_shell(self, tmpdir: str, code: str):
        script_file = os.path.join(tmpdir, "script.sh")
        with open(script_file, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(code)
        os.chmod(script_file, 0o755)  # Make the file executable
        try:
            timeout = self.config.get('execution_timeout', 30)
            if self.verbose:
                print(f"Executing shell code with timeout of {timeout} seconds...")
            result = subprocess.run([script_file], capture_output=True, text=True, timeout=timeout)
            print(result.stdout + result.stderr)
        except subprocess.TimeoutExpired:
            print(f"Execution timed out after {timeout} seconds.")
        except Exception as e:
            print(f"An error occurred during execution: {e}")

    def do_search_web(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        "Search the web for the query. Possible to get json output as well."
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for result in soup.find_all('div', class_='result__body')[:num_results]:
            title = result.find('a', class_='result__a').text
            snippet = result.find('a', class_='result__snippet').text
            link = result.find('a', class_='result__a')['href']
            results.append({"title": title, "snippet": snippet, "link": link})
        print(results)

    @alias("!")
    def do_shell(self, command: str):
        """Execute a shell command."""
        if not command:
            print("No command provided.")
            return
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = result.stdout + result.stderr
            print(output)
        except Exception as e:
            print(f"An error occurred while executing the shell command: {e}")
    
    def do_add_file(self, input_data: str):
        "Add the [file] to current context"
        if os.path.isfile(input_data):
            with open(input_data, 'r') as f:
                file_content = f.read()
        else:
            print("Not a valid file_path")
        if not self.context:
            self.context = {}
        self.context['file'] += f"\n{self.fence}\nFILE_NAME:{input_data}\n{file_content}\n{self.fence}"
        print(f"File '{input_data}' added to context.")

    def do_save_context(self, input_data: str):
        "Save the input to context"
        if not self.context:
            self.context = ""
        self.context = input_data

    def do_get_context(self, input_data: str):
        "Get information in current context"
        ctx = getattr(self, 'context', "No context available.")
        print(ctx)

    @requires_confirmation
    def do_exec_python(self, input_data: str):
        "Execute python code in context or in stdin."
        if os.path.isfile(input_data):
            with open(input_data, 'r') as f:
                code = f.read()
        else:
            code = input_data
        self.execute_code(code, "python")
    
    @requires_confirmation
    def do_exec_shell(self, input_data: str):
        "Execute shell code in context or in stdin"
        if os.path.isfile(input_data):
            with open(input_data, 'r') as f:
                code = f.read()
        else:
            code = input_data
        self.execute_code(code, "shell")

    # DOCS: https://docs.python.org/3/library/cmd.html#cmd.Cmd.completedefault
    def completedefault(self, text, line, begidx, endidx):
        """General completion method for all commands."""
        # Get all command name
        commands = [cmd[3:] for cmd in self.get_names() if cmd.startswith("do_")]

        if begidx == 0:
            # If we're at the start of the line, suggest commands
            if not text:
                # No input, return all commands
                return commands
            else:
                # Filter commands based on input
                return [cmd for cmd in commands if cmd.startswith(text)]
        else:
            # If completing arguments for a specific command
            cmd_name = line.split()[0]
            if hasattr(self, f"complete_{cmd_name}"):
                # Use specific complete_<command> if defined
                return getattr(self, f"complete_{cmd_name}")(text, line, begidx, endidx)
            else:
                # Default to no suggestions for arguments
                return []

    def onecmd(self, line):
        """Handle pipes and execute the commands in sequence."""
        # Split commands based on the pipe (|)
        commands = [cmd.strip() for cmd in line.split('|')]
        # Input/output for piping
        input_data = None

        for command in commands:
            # Split the command and its arguments
            cmd_name, *cmd_args = command.split()
            cmd_args = ' '.join(cmd_args)  # Convert list back to string
            # If there is previous output (from a pipe), pass it as input
            if input_data is not None:
                # Replace standard input with previous command's output
                cmd_args = cmd_args + input_data

            # Handle '?' explicitly as a help request
            if cmd_name == "?":
                self.do_help(cmd_args)
                continue

            # Check if the command exists
            method = getattr(self, f"do_{cmd_name}", None)
            if not method:
                # If no direct match, check for aliases
                for attr in dir(self):
                    potential_method = getattr(self, attr)
                    if callable(potential_method) and hasattr(potential_method, "aliases"):
                        if cmd_name in potential_method.aliases:
                            method = potential_method
                            break
            
            if not method:
                print(f"Unknown command: {cmd_name}")
                return False
            # Check if the command requires confirmation
            if getattr(method, "requires_confirmation", False):
                if not self.confirm_execution():
                    print(f"Command '{cmd_name}' aborted by user.")
                    return False
            
            # Redirect standard output to a string buffer to capture output
            output_buffer = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = output_buffer
            # Execute the command
            try:
                stop = method(cmd_args)
            except Exception as e:
                print(f"Error executing command {cmd_name}:{e}")
                return False
            finally:
                # Restore original stdout
                sys.stdout = old_stdout
            # Capture the output for piping to the next command
            input_data = output_buffer.getvalue().strip()
            output_buffer.close()
            if stop:
                return stop
        # Print the final output after all piped commands are executed
        if input_data:
            print(input_data)
        self.history.append((line, input_data))
        return False

    def do_exit(self, arg):
        """Exit the program."""
        print("Exiting GenSh. Goodbye!")
        self.db_conn.close()
        return True

    def load_templates(self) -> Dict[str, str]:
        template_dir = self.config.get('template_dir', 'templates')
        templates = {}
        sys_templates_dir = Path(__file__).parent / "templates"
        # Iterate through all files in the templates directory
        for template_file in sys_templates_dir.iterdir():
            if template_file.is_file() and template_file.suffix == '.yml':
                with open(sys_templates_dir / template_file.name, "r") as file:
                    templates[template_file.stem] = yaml.safe_load(file)
        if os.path.exists(template_dir):
            for filename in os.listdir(template_dir):
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    template_name = os.path.splitext(filename)[0]
                    with open(os.path.join(template_dir, filename), 'r') as f:
                        templates[template_name] = yaml.safe_load(f)
        return templates

    def do_list_templates(self, arg):
        """List all available templates."""
        print("Available templates:")
        for template_name in self.templates:
            print(f"- {template_name}")

    def get_prompt_template(self, arg):
        pattern = arg.strip()
        matches = [name for name in self.templates if name.startswith(pattern)]
        if not matches:
            raise ValueError(f"No templates found matching pattern: {pattern}")
        key = matches[0]
        return self.templates[key]

    def do_show_template(self, arg):
        """Show the content of a specific template matching the pattern."""
        pattern = arg.strip()
        if not pattern:
            print("Error: must provide a pattern");
            return
        matches = [name for name in self.templates if name.startswith(pattern)]
        if matches:
            for match in matches:
                print(f"Template {match}")
                print(yaml.dump(self.templates[match], default_flow_style=False))
        else:
            print(f"Template '{arg}' not found.")

    def do_history(self, arg):
        """Show command history."""
        for i, (cmd, result) in enumerate(self.history, 1):
            print(f"{i}. Command: {cmd}")
            print(f"   Result: {result[:50]}..." if len(result) > 50 else f"   Result: {result}")
            print()

    def do_set_model(self, arg):
        """Set the AI model to use (e.g., gpt-3.5-turbo, claude, or an Ollama model name)."""
        self.config['model'] = arg
        print(f"Model set to: {arg}")

    def do_show_config(self, arg):
        """Show the current configuration."""
        print("Current configuration:")
        for key, value in self.config.items():
            print(f"{key}: {value}")

    def do_set_config(self, arg):
        """Set a configuration value. Usage: set_config key value"""
        parts = arg.split(maxsplit=1)
        if len(parts) == 2:
            key, value = parts
            self.config[key] = value
            print(f"Configuration updated: {key} = {value}")
        else:
            print("Usage: set_config key value")

def load_config(config_file: str) -> Dict[str, Any]:
    config_path = Path(config_file)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def main():
    import argparse
    ver = importlib.metadata.version("gensh") 
    parser = argparse.ArgumentParser(description="GenSh - Generative AI Toolkit Shell")
    parser.add_argument('--config', default='~/.gensh_config.json', help='Path to configuration file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('-l', '--list', metavar="PATTERN", type=str, help="Show templates matching the pattern")
    parser.add_argument('-c', '--command', metavar="COMMAND", type=str, help="GenSh Command with pipeline")
    parser.add_argument('-p', '--pattern', metavar=("PATTERN"), nargs='*', help="Execute AI prompt from pattern files")
    parser.add_argument('--patterns-dir', default='~/.gensh/fabric/patterns', help='Directory containing pattern files')
    parser.add_argument('--version', action='version', version=f'%(prog)s {ver}')
    args = parser.parse_args()

    config = load_config(os.path.expanduser(args.config))
    shell = GenShell(ver, config, verbose=args.verbose)
    if args.list:
        shell.do_show_template(args.list)
    if args.command:
        shell.onecmd(args.command)
    elif args.pattern:
        prompt = ' '.join(args.pattern[1:])
        shell.do_exec_pattern(f"{args.pattern[0]} '{prompt}'")
    else:
        import readline, rlcompleter
        # Ensure tab completion works
        readline.parse_and_bind("tab: complete")
        # Check if readline is available
        if "libedit" in readline.__doc__:
            readline.parse_and_bind("bind ^I rl_complete")
        shell.cmdloop()

if __name__ == "__main__":
    main()
