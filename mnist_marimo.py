import pathlib
import subprocess
import marimo as mo
import yaml
import wandb

app = mo.App(width="full")


def _run_command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = "".join(
        [
            "$ ",
            " ".join(command),
            "\n",
            result.stdout,
            result.stderr,
        ]
    )
    return output.strip()


@app.cell
def __():
    mo.md(
        r"""
# Simple MNIST Digit Recognizer (Interactive)

This marimo notebook turns the README into an interactive walkthrough. You can configure
hyperparameters, log into Weights & Biases (W&B), train the model, and serve predictions
with the API + web UI.
"""
    )
    return


@app.cell
def __():
    mo.md(
        r"""
## 1. Install Dependencies

This project uses [uv](https://docs.astral.sh/uv/) for fast dependency management. If you
haven't installed it yet, run one of the commands below in a terminal:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```
"""
    )
    return


@app.cell
def __():
    mo.md("Click to install dependencies via `uv sync`.")
    install_button = mo.ui.button(label="Run uv sync")

    output = None
    if install_button.value:
        output = _run_command(["uv", "sync"])

    if output:
        mo.md(f"```\n{output}\n```")

    return install_button, output


@app.cell
def __():
    mo.md(
        r"""
## 2. Configure Hyperparameters

Edit `config.yaml` below and save it. The values you set here control training epochs,
learning rate, and batch size.
"""
    )
    return


@app.cell
def __():
    config_path = pathlib.Path("config.yaml")
    config_text = config_path.read_text()

    config_editor = mo.ui.code_editor(
        value=config_text,
        language="yaml",
        label="config.yaml",
    )
    save_button = mo.ui.button(label="Save config.yaml")
    save_status = None

    if save_button.value:
        try:
            yaml.safe_load(config_editor.value)
            config_path.write_text(config_editor.value)
            save_status = "✅ Saved config.yaml"
        except yaml.YAMLError as exc:
            save_status = f"❌ YAML error: {exc}"

    if save_status:
        mo.md(save_status)

    return config_editor, save_button, save_status


@app.cell
def __():
    mo.md(
        r"""
## 3. Log in to Weights & Biases (W&B)

Paste your API key to log in. You can find it at
[https://wandb.ai/authorize](https://wandb.ai/authorize).
"""
    )
    return


@app.cell
def __():
    api_key_input = mo.ui.text(label="W&B API Key (optional)")
    login_button = mo.ui.button(label="Login to W&B")
    login_status = None

    if login_button.value:
        key = api_key_input.value.strip() or None
        try:
            wandb.login(key=key, relogin=True)
            login_status = "✅ Logged in to W&B."
        except wandb.errors.UsageError as exc:
            login_status = f"❌ W&B login failed: {exc}"

    if login_status:
        mo.md(login_status)

    return api_key_input, login_button, login_status


@app.cell
def __():
    mo.md(
        r"""
## 4. Train the Model

Click to run `python train.py`. The training script logs metrics, tables, and a model
artifact to W&B. The terminal output includes a link to the run page.
"""
    )
    return


@app.cell
def __():
    train_button = mo.ui.button(label="Run training")
    train_output = None

    if train_button.value:
        train_output = _run_command(["uv", "run", "python", "train.py"])

    if train_output:
        mo.md(f"```\n{train_output}\n```")

    return train_button, train_output


@app.cell
def __():
    mo.md(
        r"""
## 5. View Metrics & Artifacts

After training, open the W&B run link printed in the output. You'll find:

- `batch_loss`, `epoch_loss`, `epoch_accuracy`, `test_accuracy`
- `sample_test_predictions` and `misclassified_test_examples` tables
- The `mnist-simple-nn` model artifact
"""
    )
    return


@app.cell
def __():
    mo.md(
        r"""
## 6. Serve the Model (API + Web UI)

Start the FastAPI server in another terminal:

```bash
uv run uvicorn serve:app
```

Then open the web UI in your browser:

- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

The page lets you upload an image, preview it, and request a prediction.
"""
    )
    return


@app.cell
def __():
    mo.md(
        r"""
## 7. Experiment Systematically

Try adjusting **one** hyperparameter at a time in `config.yaml` and re-running training.
Use descriptive run names in W&B if you want to track experiments easily.
"""
    )
    return


if __name__ == "__main__":
    app.run()
