# ShellSage

This is a fork of shell_sage modified to use your local ollama models.
 
## Usage

### Installation

Install latest from the GitHub
[repository](https://github.com/AnswerDotAI/shell_sage):

or from [pypi](https://pypi.org/project/shell_sage/)

```sh
pip install shell_sage
```

We recommend also setting up your terminal editor of choice to keep the editor content's displayed in the terminal on exit. This allows `ShellSage` to see the files you have been working on. Here is how you can do this in vim:

```sh
echo "set t_ti= t_te=" >> ~/.vimrc
```

You will also need to get an Anthropic API key and set its environment variable:

```sh
export OLLAMA_MODEL="qwen2.5-coder:7b"
```

## How to use

`ShellSage` is designed to be ran inside a tmux session since it relies on tmux for getting what has is displayed on your terminal as context. If you don't want to use tmux, you will need to use the `--NH` command, which will not include your terminal history.

```sh
ssage hi ShellSage
```

```markdown
Hello! I'm ShellSage, your command-line assistant. I can help you with:

- Bash commands and scripting
- System administration tasks
- Git operations
- File management
- Process handling
- And more!
```

You can also pipe outputs into `ShellSage`:

```sh
cat file.txt | ssage summarize this file
```
