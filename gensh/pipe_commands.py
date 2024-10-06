import os
import abc

class PipeCommand(abc.ABC):
    @abc.abstractmethod
    def execute(self, input_data: str, shell: 'GenShell') -> str:
        pass

    @abc.abstractmethod
    def help(self) -> str:
        pass

class GenerateCodeCommand(PipeCommand):
    def execute(self, input_data: str, shell: 'GenShell') -> str:
        return shell.generate_code(input_data, "python")

    def help(self) -> str:
        return "Generate code based on the given input."

class AddFileToContextCommand(PipeCommand):
    def execute(self, input_data: str, shell: 'GenShell') -> str:
        if os.path.isfile(input_data):
            with open(input_data, 'r') as f:
                file_content = f.read()
        else:
            return "Not a valid file_path"
        if 'context' not in shell.__dict__:
            shell.context = ""
        shell.context += f"\n{shell.fence}\nFILE_NAME:{input_data}\n{file_content}\n{shell.fence}"
        return f"File '{input_data}' added to context."

    def help(self) -> str:
        return "Add a file to the current context."

class ShowContextCommand(PipeCommand):
    def execute(self, input_data: str, shell: 'GenShell') -> str:
        return getattr(shell, 'context', "No context available.")

    def help(self) -> str:
        return "Show the current context."

class ExecPythonCommand(PipeCommand):
    def execute(self, input_data: str, shell: 'GenShell') -> str:
        if os.path.isfile(input_data):
            with open(input_data, 'r') as f:
                code = f.read()
        else:
            code = input_data

        if shell.confirm_execution():
            return shell.execute_code(code, "python")
        return "Code execution cancelled."

    def help(self) -> str:
        return "Execute Python code."

class ExecShellCommand(PipeCommand):
    def execute(self, input_data: str, shell: 'GenShell') -> str:
        if os.path.isfile(input_data):
            with open(input_data, 'r') as f:
                code = f.read()
        else:
            code = input_data
        
        if shell.confirm_execution():
            return shell.execute_code(code, "shell")
        return "Code execution cancelled."

    def help(self) -> str:
        return "Execute shell commands."

class SearchWebCommand(PipeCommand):
    def execute(self, input_data: str, shell: 'GenShell') -> str:
        options = input_data.split()
        query = options[0]
        output_format = "text"
        if "--output" in options:
            output_format = options[options.index("--output") + 1]
        search_results = shell.search_web(query)
        if output_format == "json":
            import json
            return json.dumps(search_results)
        else:
            return "\n".join([f"{r['title']}: {r['snippet']}" for r in search_results])

    def help(self) -> str:
        return "Search the web for the given query."
