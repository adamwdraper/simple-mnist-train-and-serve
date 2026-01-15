import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    # Main title and summary
    intro_header = mo.md(
        """
        # Simple MNIST Digit Recognizer

        Welcome! This interactive notebook will teach you the fundamentals of training a neural network 
        by building a model that recognizes handwritten digits. You'll train a real model, track your 
        experiments with **Weights & Biases**, and see how changing settings affects performance.
        """
    )

    # Accordion content for deeper explanations
    what_is_mnist = mo.md(
        """
        **MNIST** (Modified National Institute of Standards and Technology) is the "Hello World" of 
        machine learning. It's a dataset of 70,000 handwritten digit images that has been used since 
        1998 to benchmark image classification algorithms.

        **Why MNIST matters:**
        - It's simple enough to train quickly (minutes, not hours)
        - Complex enough to demonstrate real ML concepts
        - Small enough to run on any computer (no GPU required)
        - Well-understood, so you can compare your results to others

        **The dataset contains:**
        - **60,000 training images** - used to teach the model
        - **10,000 test images** - used to evaluate performance on unseen data
        - Each image is **28x28 pixels**, grayscale (single channel)
        - Labels are digits **0-9**

        The images were collected from Census Bureau employees and high school students writing digits.
        """
    )

    what_is_classification = mo.md(
        """
        **Image classification** is teaching a computer to look at an image and assign it to a category.

        **The problem we're solving:**
        - **Input**: A 28x28 pixel image of a handwritten digit
        - **Output**: A prediction of which digit (0-9) it represents

        This is a **supervised learning** task because we have labeled examples (images paired with 
        the correct digit) that we use to train the model.

        **How the model learns:**
        1. **See an image** - The model receives pixel values as input
        2. **Make a guess** - It outputs probabilities for each digit (0-9)
        3. **Check the answer** - Compare the guess to the true label
        4. **Adjust weights** - If wrong, tweak internal parameters to do better next time
        5. **Repeat** - Process thousands of images, gradually improving

        After training, the model can classify new images it has never seen before.
        """
    )

    what_is_neural_network = mo.md(
        """
        A **neural network** is a computational model inspired by how the brain processes information.

        **Our SimpleNN architecture:**

        ```
        Input (784 pixels) → Hidden Layer (128 neurons) → Output (10 classes)
        ```

        **How it works:**

        1. **Input Layer**: The 28x28 image is flattened into 784 numbers (pixel values)

        2. **Hidden Layer**: 128 neurons, each connected to all 784 inputs
           - Each neuron computes a weighted sum of inputs plus a bias
           - A **ReLU activation** (Rectified Linear Unit) is applied: negative values become 0
           - This layer learns to detect patterns like edges, curves, and shapes

        3. **Output Layer**: 10 neurons, one for each digit class
           - Each outputs a "score" for how likely the image is that digit
           - The highest score is the model's prediction

        **What gets learned:**
        - The **weights** (128 × 784 + 10 × 128 = 101,632 parameters!)
        - These start random and are adjusted during training
        """
    )

    what_is_wandb = mo.md(
        """
        **Weights & Biases (W&B)** is an experiment tracking platform for machine learning.

        **Why use it:**
        - **Track metrics** - See loss and accuracy over time as charts
        - **Compare runs** - Run multiple experiments and compare them side-by-side
        - **Log artifacts** - Save your trained models for later use
        - **Reproduce results** - Every run logs its configuration

        **What we'll log:**
        - `batch_loss` - How wrong the model is on each batch of images
        - `epoch_loss` - Average loss over a full pass through the training data
        - `epoch_accuracy` - Percentage of training images classified correctly
        - `test_accuracy` - Final accuracy on unseen test images
        - `sample_predictions` - Actual images with predicted vs true labels
        - `misclassified_examples` - Images where the model got it wrong

        After training, you'll get a link to your W&B dashboard to explore all of this visually.
        """
    )

    mo.vstack([
        intro_header,
        mo.accordion({
            "What is MNIST?": what_is_mnist,
            "What is image classification?": what_is_classification,
            "How does a neural network work?": what_is_neural_network,
            "What is Weights & Biases?": what_is_wandb,
        }),
        mo.md("---\n\n**New to W&B?** [Sign up here](https://wandb.ai/site) to get your API key."),
    ])
    return


@app.cell
def _():
    # Core imports
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from sklearn.metrics import accuracy_score
    import wandb
    import numpy as np
    from PIL import Image
    import io
    import yaml
    import subprocess
    return (
        Image,
        accuracy_score,
        io,
        nn,
        np,
        optim,
        subprocess,
        torch,
        torchvision,
        transforms,
        wandb,
        yaml,
    )


@app.cell
def _(nn):
    # Model definition (same as model.py)
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    return (SimpleNN,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 1: Authenticate with Weights & Biases

        Enter your W&B API key below to enable experiment tracking. 
        You can find your API key at [wandb.ai/authorize](https://wandb.ai/authorize).
        """
    )
    return


@app.cell
def _(mo):
    api_key_input = mo.ui.text(
        placeholder="Paste your W&B API key here...",
        kind="password",
        label="W&B API Key",
        full_width=True,
    )
    login_button = mo.ui.run_button(label="Login to W&B")
    return api_key_input, login_button


@app.cell
def _(api_key_input, login_button, mo):
    mo.hstack([api_key_input, login_button], justify="start", gap=1)
    return


@app.cell
def _(api_key_input, login_button, mo, wandb):
    login_status = None
    logged_in = False

    if login_button.value and api_key_input.value:
        try:
            wandb.login(key=api_key_input.value, relogin=True)
            login_status = mo.callout(
                mo.md("**Successfully logged in to W&B!** You can now run training."),
                kind="success",
            )
            logged_in = True
        except Exception as e:
            login_status = mo.callout(
                mo.md(f"**Login failed:** {str(e)}"),
                kind="danger",
            )
            logged_in = False
    elif login_button.value and not api_key_input.value:
        login_status = mo.callout(
            mo.md("Please enter your API key before clicking Login."),
            kind="warn",
        )
        logged_in = False

    login_status if login_status else mo.md("_Click 'Login to W&B' after entering your API key._")
    return logged_in, login_status


@app.cell
def _(mo):
    mo.md("## Step 2: Configure Hyperparameters")
    return


@app.cell
def _(mo):
    epochs_slider = mo.ui.slider(
        start=1,
        stop=20,
        value=5,
        step=1,
        label="Epochs",
        show_value=True,
    )

    epochs_explanation = mo.md(
        """
        **What is an epoch?**

        One epoch = one complete pass through all 60,000 training images.

        **Effects:**
        - **Too few (1-2)**: Model hasn't learned enough patterns → **underfitting** (low accuracy)
        - **Too many (20+)**: Model memorizes training data instead of learning general patterns → **overfitting**
        - **Signs of overfitting**: Training accuracy keeps improving, but test accuracy stops improving or gets worse

        **Try:** Start with 5 epochs, then try 10 to see if accuracy improves.
        """
    )
    return epochs_explanation, epochs_slider


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(
        start=0.0001,
        stop=0.1,
        value=0.001,
        step=0.0001,
        label="Learning Rate",
        show_value=True,
    )

    lr_explanation = mo.md(
        """
        **What is learning rate?**

        The learning rate controls how much the model adjusts its weights after each batch. Think of it 
        as the "step size" when walking downhill to find the lowest point (best model).

        **Effects:**
        - **Too high (0.01-0.1)**: Model takes huge steps, might overshoot the best solution → loss jumps around or explodes
        - **Too low (0.0001)**: Model takes tiny steps, learns very slowly → needs many more epochs
        - **Just right (0.001)**: Balanced learning, steady improvement

        **Try:** The default 0.001 works well for most cases. Try 0.01 to see unstable training.
        """
    )
    return lr_explanation, lr_slider


@app.cell
def _(mo):
    batch_size_dropdown = mo.ui.dropdown(
        options={"32": 32, "64": 64, "128": 128, "256": 256},
        value="64",
        label="Batch Size",
    )

    batch_explanation = mo.md(
        """
        **What is batch size?**

        Instead of updating weights after every single image, we process images in batches. The batch 
        size is how many images we look at before making one weight update.

        **Effects:**
        - **Smaller (32)**: More frequent updates, "noisier" learning, can escape bad local minima. Uses less memory. Slower per epoch.
        - **Larger (256)**: Fewer updates per epoch, smoother learning, faster training. Uses more memory. May converge to worse solutions.

        **Math:** With 60,000 images and batch_size=64, you get 938 weight updates per epoch.

        **Try:** 64 is a good default. Try 32 for potentially better generalization, 256 for faster training.
        """
    )
    return batch_explanation, batch_size_dropdown


@app.cell
def _(batch_explanation, batch_size_dropdown, epochs_explanation, epochs_slider, lr_explanation, lr_slider, mo):
    # Three columns: each has slider on top, explanation below
    mo.hstack([
        mo.vstack([
            mo.md("**Epochs**"),
            epochs_slider,
            epochs_explanation,
        ], align="start"),
        mo.vstack([
            mo.md("**Learning Rate**"),
            lr_slider,
            lr_explanation,
        ], align="start"),
        mo.vstack([
            mo.md("**Batch Size**"),
            batch_size_dropdown,
            batch_explanation,
        ], align="start"),
    ], justify="start", gap=3, align="start")
    return


@app.cell
def _(batch_size_dropdown, epochs_slider, lr_slider, mo, yaml):
    # Sync hyperparameters to config.yaml so train.py uses them
    _config = {
        "epochs": epochs_slider.value,
        "lr": lr_slider.value,
        "batch_size": int(batch_size_dropdown.value),
    }
    with open("config.yaml", "w") as _f:
        yaml.dump(_config, _f, default_flow_style=False)

    config_summary = mo.callout(
        mo.md(
            f"**Current config:** {epochs_slider.value} epochs, LR={lr_slider.value:.4f}, batch size={batch_size_dropdown.value} *(saved to config.yaml)*"
        ),
        kind="info",
    )
    config_summary
    return (config_summary,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 3: Train the Model

        Click the button below to run the training script. This will:
        1. Download the MNIST dataset (if not cached)
        2. Train for the configured number of epochs
        3. Log metrics to W&B
        4. Evaluate on the test set
        5. Save the model artifact

        The console output below will show the real training progress, including the W&B run URL.
        """
    )
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Run Training Script", kind="success")
    train_button
    return (train_button,)


@app.cell
def _(logged_in, mo, subprocess, train_button):
    training_output = None

    if train_button.value:
        if not logged_in:
            training_output = mo.callout(
                mo.md("**Please log in to W&B first** (Step 1) before training."),
                kind="warn",
            )
        else:
            mo.output.replace(mo.md("**Running training script... This may take a few minutes.**"))

            # Run the training script as a subprocess
            result = subprocess.run(
                ["uv", "run", "python", "train.py"],
                capture_output=True,
                text=True,
                cwd=".",
            )

            # Combine stdout and stderr for full output
            console_output = result.stdout + result.stderr

            if result.returncode == 0:
                training_output = mo.vstack([
                    mo.callout(
                        mo.md("**Training Complete!** See the output below for your W&B run link."),
                        kind="success",
                    ),
                    mo.md("### Console Output"),
                    mo.md(f"```\n{console_output}\n```"),
                ])
            else:
                training_output = mo.vstack([
                    mo.callout(
                        mo.md("**Training failed.** See the error output below."),
                        kind="danger",
                    ),
                    mo.md("### Console Output"),
                    mo.md(f"```\n{console_output}\n```"),
                ])

            mo.output.replace(training_output)

    training_output if training_output else mo.md(
        "_Click 'Run Training Script' to begin. Make sure you're logged in to W&B first._"
    )
    return (training_output,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 4: Test the Model

        Upload an image of a handwritten digit to see what the model predicts!

        **Tips for best results:**
        - Use a clear image of a single digit (0-9)
        - White digit on dark background works best (like MNIST)
        - The image will be automatically resized to 28x28 pixels
        """
    )
    return


@app.cell
def _(mo):
    file_upload = mo.ui.file(
        filetypes=[".png", ".jpg", ".jpeg", ".gif", ".bmp"],
        label="Upload a digit image",
    )
    file_upload
    return (file_upload,)


@app.cell
def _(
    Image,
    SimpleNN,
    file_upload,
    io,
    mo,
    np,
    torch,
    transforms,
):
    prediction_result = None

    if file_upload.value and len(file_upload.value) > 0:
        uploaded_file = file_upload.value[0]
        image_bytes = uploaded_file.contents

        try:
            # Load and preprocess image
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to grayscale
            img = img.convert("L")

            # Resize to 28x28
            img = img.resize((28, 28), Image.Resampling.LANCZOS)

            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)

            # Invert if the image appears to have dark digit on light background
            # (MNIST has light digit on dark background)
            if img_array.mean() > 127:
                img_array = 255 - img_array

            # Normalize to 0-1 range
            img_array = img_array / 255.0

            # Apply MNIST normalization
            img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
            normalize = transforms.Normalize((0.1307,), (0.3081,))
            img_tensor = normalize(img_tensor)

            # Load trained model from file
            try:
                model_to_use = SimpleNN()
                model_to_use.load_state_dict(torch.load("mnist_model.pth", weights_only=True))
                model_to_use.eval()
            except FileNotFoundError:
                prediction_result = mo.callout(
                    mo.md("**No trained model found.** Please train the model first (Step 3)."),
                    kind="warn",
                )
                model_to_use = None

            if model_to_use is not None:
                model_to_use.eval()
                with torch.no_grad():
                    _outputs = model_to_use(img_tensor)
                    probabilities = torch.softmax(_outputs, dim=1).squeeze()
                    predicted_digit = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_digit].item() * 100

                # Build confidence table
                confidence_data = [
                    {"Digit": i, "Confidence": f"{probabilities[i].item() * 100:.2f}%"}
                    for i in range(10)
                ]

                # Display uploaded image (resized for display)
                display_img = Image.open(io.BytesIO(image_bytes))
                display_img = display_img.resize((112, 112), Image.Resampling.LANCZOS)

                prediction_result = mo.vstack([
                    mo.hstack([
                        mo.vstack([
                            mo.md("**Uploaded Image:**"),
                            mo.image(src=image_bytes),
                        ]),
                        mo.vstack([
                            mo.md(f"## Predicted Digit: **{predicted_digit}**"),
                            mo.md(f"Confidence: **{confidence:.1f}%**"),
                        ]),
                    ], justify="start", gap=2),
                    mo.md("### All Confidence Scores"),
                    mo.ui.table(confidence_data),
                ])

        except Exception as e:
            prediction_result = mo.callout(
                mo.md(f"**Error processing image:** {str(e)}"),
                kind="danger",
            )

    prediction_result if prediction_result else mo.md(
        "_Upload an image above to test the model._"
    )
    return (prediction_result,)


@app.cell
def _(mo):
    mo.md(
        """
        ---

        ## Experimenting with Hyperparameters

        Here are some experiments to try:

        | Experiment | Change | What to Observe |
        |------------|--------|-----------------|
        | **High Learning Rate** | Set LR to 0.01 or 0.1 | Loss may fluctuate or diverge |
        | **Low Learning Rate** | Set LR to 0.0001 | Training is slower but more stable |
        | **More Epochs** | Increase to 10-20 | Watch for overfitting (training acc up, test acc plateaus) |
        | **Fewer Epochs** | Decrease to 1-2 | Model underfits (low accuracy) |
        | **Large Batch Size** | Use 256 | Faster training, possibly less accurate |
        | **Small Batch Size** | Use 32 | Slower but potentially better generalization |

        Each training run is logged to W&B, so you can compare experiments in your dashboard!
        """
    )
    return


if __name__ == "__main__":
    app.run()
