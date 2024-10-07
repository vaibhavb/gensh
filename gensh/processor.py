import yaml
import re
import os
import subprocess
from typing import List, Dict, Any
import openai
from anthropic import Anthropic
import ollama
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
import importlib.resources

def load_api_tokens():
    load_dotenv()
    tokens = {}
    for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']:
        tokens[key] = os.getenv(key)
        if tokens[key] is None:
            tokens[key] = input(f"Please enter your {key}: ").strip()
            if not tokens[key]:
                print(f"Warning: {key} not provided. Some functionality may be limited.")
    return tokens

class GenShell(cmd.Cmd):
    prompt = "gensh> "
    fence = "---"
    defaultConfig = {
        "template_dir": "templates",
        "model": "gpt-3.5-turbo",
        "execution_timeout": 30,
        "max_retries": 3,
        "retry_delay": 1,
        "db_path": "gensh_logs.db",
        "output_format": "text"
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
        self.anthropic_client = Anthropic(api_key=self.api_tokens['ANTHROPIC_API_KEY'])
        self.context = None
        self.session_id = str(uuid.uuid4())
        self.db_conn = self.init_database()
        self.templates = self.load_templates()
        self.history = []
        self.intro = f"Welcome to GenSh {self.version}. Type help or ? to list commands.\n"

    def init_database(self):
        db_path = self.config.get('db_path', 'gensh_logs.db')
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

    def do_generate_code(self, query: str, execution_type: str = 'python') -> str:
        "Generate code. If query starts as 'use template architect' then it will use 'architect' template, else default to python code generator."
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
                user_prompt = f"Generate {execution_type} code based on the template. {template_args}"
            else:
                print(f"Template '{template_name}' not found.")
                return None
        else:
            query = self.replace_context_variables_in_query(query)
            system_prompt = f"You are a {execution_type} code generator. Generate concise, efficient code without explanations or markdown formatting."
            user_prompt = f"Generate {execution_type} code to {query}. Provide only the code, no explanations or markdown formatting."

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
                
                print(f"Generated {execution_type} code:")
                print(f"```{execution_type}")
                print(code)
                print("```")
            except Exception as e:
                if self.verbose:
                    print(f"Error generating code: {e}")
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to generate code after {max_retries} attempts.")

    def strip_markdown_code_block(self, code: str) -> str:
        code = re.sub(r'^```[\w]*\n', '', code)
        code = re.sub(r'\n```$', '', code)
        return code.strip()

    def confirm_execution(self) -> bool:
        while True:
            response = input("Do you want to execute this code? (yes/no): ").lower()
            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                return False
            else:
                print("Please answer 'yes' or 'no'.")

    def execute_code(self, code: str, execution_type: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            if execution_type == "python":
                output = self.execute_python(tmpdir, code)
            elif execution_type == "shell":
                output = self.execute_shell(tmpdir, code)
            return output

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
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return f"Execution timed out after {timeout} seconds."
        except Exception as e:
            return f"An error occurred during execution: {e}"

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

    def do_exec_python(self, input_data: str):
        "Execute python code in context or in stdin."
        if os.path.isfile(input_data):
            with open(input_data, 'r') as f:
                code = f.read()
        else:
            code = input_data
        if self.confirm_execution():
            return self.execute_code(code, "python")
        print("Code execution cancelled.")

    def do_exec_shell(self, input_data: str):
        "Execute shell code in context or in stdin"
        if os.path.isfile(input_data):
            with open(input_data, 'r') as f:
                code = f.read()
        else:
            code = input_data
        if self.confirm_execution():
            return self.execute_code(code, "shell")
        print("Code execution cancelled.")

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
            # Redirect standard output to a string buffer to capture output
            output_buffer = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = output_buffer
            # Execute the command
            try:
                stop = super().onecmd(f"{cmd_name} {cmd_args}") 
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

    def do_exit(self, arg):
        """Exit the program."""
        print("Exiting GenSh. Goodbye!")
        self.db_conn.close()
        return True

    def load_templates(self) -> Dict[str, str]:
        template_dir = self.config.get('template_dir', 'templates')
        templates = {}
        with importlib.resources.files('templates') as sys_templates_dir:
            # Iterate through all files in the templates directory
            for template_file in sys_templates_dir.iterdir():
                if template_file.is_file() and template_file.suffix == '.yml':
                    with importlib.resources.open_text('templates', template_file.name) as file:
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

    def do_show_template(self, arg):
        """Show the content of a specific template."""
        if arg in self.templates:
            print(yaml.dump(self.templates[arg], default_flow_style=False))
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
    ver = f"{__import__("gensh").__version__}"
    parser = argparse.ArgumentParser(description="GenSh - Generate and execute code using natural language")
    parser.add_argument('--config', default='~/.gensh_config.json', help='Path to configuration file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('--version', action='version', version=f'%(prog)s {ver}')
    args = parser.parse_args()

    config = load_config(os.path.expanduser(args.config))
    shell = GenShell(ver, config, verbose=args.verbose)
    shell.cmdloop()

if __name__ == "__main__":
    main()
