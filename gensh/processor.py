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

from .pipe_commands import PipeCommand, GenerateCodeCommand, ExecPythonCommand, ExecShellCommand, SearchWebCommand

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
    intro = "Welcome to GenSh. Type help or ? to list commands.\n"
    prompt = "gensh> "

    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        super().__init__()
        self.config = config
        self.verbose = verbose
        self.api_tokens = load_api_tokens()
        self.openai_client = openai.OpenAI(api_key=self.api_tokens['OPENAI_API_KEY'])
        self.anthropic_client = Anthropic(api_key=self.api_tokens['ANTHROPIC_API_KEY'])
        self.last_output = None
        self.session_id = str(uuid.uuid4())
        self.db_conn = self.init_database()
        self.pipe_commands = {}
        self.register_pipe_command('generate', GenerateCodeCommand())
        self.register_pipe_command('exec_python', ExecPythonCommand())
        self.register_pipe_command('exec_shell', ExecShellCommand())
        self.register_pipe_command('search_web', SearchWebCommand())
        self.templates = self.load_templates()
        self.history = []

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

    def generate_code(self, query: str, execution_type: str) -> str:
        template_match = re.match(r'use template (\w+)(.*)', query, re.IGNORECASE)
        if template_match:
            template_name = template_match.group(1)
            template_args = template_match.group(2).strip()
            if template_name in self.templates:
                template = self.templates[template_name]
                prompt = template['prompt']
                for var, desc in template['variables'].items():
                    value = input(f"Enter {desc}: ")
                    prompt = prompt.replace(f"{{{var}}}", value)
                system_prompt = f"You are a {execution_type} code generator. Use the following template to generate code:\n\n{prompt}"
                user_prompt = f"Generate {execution_type} code based on the template. {template_args}"
            else:
                print(f"Template '{template_name}' not found.")
                return None
        else:
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
                return code

            except Exception as e:
                if self.verbose:
                    print(f"Error generating code: {e}")
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to generate code after {max_retries} attempts.")
                    return None

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
            self.last_output = output
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
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return f"Execution timed out after {timeout} seconds."
        except Exception as e:
            return f"An error occurred during execution: {e}"

    def search_web(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for result in soup.find_all('div', class_='result__body')[:num_results]:
            title = result.find('a', class_='result__a').text
            snippet = result.find('a', class_='result__snippet').text
            link = result.find('a', class_='result__a')['href']
            results.append({"title": title, "snippet": snippet, "link": link})
        return results

    def process_command(self, line):
        commands = line.split("|")
        result = None
        for command in commands:
            command = command.strip()
            if command.startswith('"') and command.endswith('"'):
                query = command[1:-1]
                if result:
                    query = query.replace("$result", result)
                result = self.pipe_commands['generate'].execute(query, self)
            else:
                cmd_parts = command.split(maxsplit=1)
                cmd_name = cmd_parts[0]
                cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""
                if cmd_name in self.pipe_commands:
                    result = self.pipe_commands[cmd_name].execute(cmd_args or result, self)
                else:
                    print(f"Unknown command: {command}")
                    return
        if result:
            print(result)
        self.last_output = result
        self.history.append((line, result))

    def default(self, line):
        self.process_command(line)

    def do_exit(self, arg):
        """Exit the program."""
        print("Exiting GenSh. Goodbye!")
        self.db_conn.close()
        return True

    def register_pipe_command(self, name: str, command: PipeCommand):
        self.pipe_commands[name] = command

    def load_templates(self) -> Dict[str, str]:
        template_dir = self.config.get('template_dir', 'templates')
        templates = {}
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

    def do_help_pipe(self, arg):
        """Show help for pipe commands."""
        if arg:
            if arg in self.pipe_commands:
                print(f"Help for {arg}:")
                print(self.pipe_commands[arg].help())
            else:
                print(f"Unknown pipe command: {arg}")
        else:
            print("Available pipe commands:")
            for cmd_name, cmd in self.pipe_commands.items():
                print(f"- {cmd_name}: {cmd.help()}")

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
    parser = argparse.ArgumentParser(description="GenSh - Generate and execute code using natural language")
    parser.add_argument('--config', default='~/.gensh_config.json', help='Path to configuration file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__import__("gensh").__version__}')
    args = parser.parse_args()

    config = load_config(os.path.expanduser(args.config))
    shell = GenShell(config, verbose=args.verbose)
    shell.cmdloop()

if __name__ == "__main__":
    main()
