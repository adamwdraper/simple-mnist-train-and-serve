import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


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
    import re
    import yaml
    import subprocess
    return (
        Image,
        accuracy_score,
        io,
        nn,
        np,
        optim,
        re,
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


# ============================================================================
# TAB 1: Project Introduction
# ============================================================================

@app.cell
def _(mo):
    intro_content = mo.vstack([
        mo.md(
            """
            ##

            This interactive notebook teaches you the fundamentals of training a neural network 
            by building a model that recognizes handwritten digits. You'll train a real model, track your 
            experiments with **Weights & Biases**, and see how changing settings affects performance.
            
            We'll be using **MNIST**, the "Hello World" of machine learning — a dataset of 70,000 handwritten 
            digit images (60,000 for training, 10,000 for testing). Each image is 28x28 pixels, and the goal 
            is to classify which digit (0-9) it represents. It's simple enough to train in minutes, yet 
            complex enough to demonstrate real ML concepts.

            ---
            
            **👆 Use the tabs above to navigate through the steps!**
            """
        ),
    ])
    return (intro_content,)


# ============================================================================
# TAB 2: W&B Login - UI Elements
# ============================================================================

@app.cell
def _(wandb):
    # Check if user is already logged in via saved credentials in ~/.netrc
    existing_api_key = wandb.api.api_key or ""
    already_logged_in = existing_api_key != ""
    return already_logged_in, existing_api_key


@app.cell
def _(existing_api_key, mo):
    api_key_input = mo.ui.text(
        value=existing_api_key,
        placeholder="Paste your W&B API key here...",
        kind="password",
        label="W&B API Key",
        full_width=True,
    )
    login_button = mo.ui.run_button(label="Login to W&B")
    return api_key_input, login_button


@app.cell
def _(already_logged_in, api_key_input, login_button, mo, wandb):
    login_status = None
    logged_in = already_logged_in

    if already_logged_in and not login_button.value:
        # User is already authenticated from a previous session
        login_status = mo.callout(
            mo.md("**Already logged in to W&B!** Your saved credentials are being used. You can proceed to training."),
            kind="success",
        )
    elif login_button.value and api_key_input.value:
        try:
            wandb.login(key=api_key_input.value, relogin=True)
            login_status = mo.callout(
                mo.md("**Successfully logged in to W&B!** Your credentials have been saved for future sessions."),
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

    return logged_in, login_status


@app.cell
def _(api_key_input, login_button, login_status, mo):
    login_items = [
        mo.md(
            """
            ##

            Enter your W&B API key below to enable experiment tracking. 
            You can find your API key at [wandb.ai/authorize](https://wandb.ai/authorize).
            """
        ),
        api_key_input,
        login_button,
    ]
    
    if login_status:
        login_items.append(login_status)
    
    login_content = mo.vstack(login_items)
    return (login_content,)


# ============================================================================
# TAB 3: Training (Hyperparameters + Run)
# ============================================================================

@app.cell
def _(mo):
    epochs_slider = mo.ui.slider(
        start=1,
        stop=20,
        value=5,
        step=1,
        label="Epochs",
        show_value=False,
    )
    return (epochs_slider,)


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(
        start=0.0001,
        stop=0.01,
        value=0.001,
        step=0.0001,
        label="Learning Rate",
        show_value=False,
    )
    return (lr_slider,)


@app.cell
def _(mo):
    batch_size_dropdown = mo.ui.dropdown(
        options={"32": 32, "64": 64, "128": 128, "256": 256},
        value="64",
        label="Batch Size",
    )
    return (batch_size_dropdown,)


@app.cell
def _(batch_size_dropdown, epochs_slider, lr_slider, yaml):
    # Sync hyperparameters to config.yaml
    _config = {
        "epochs": epochs_slider.value,
        "lr": lr_slider.value,
        "batch_size": int(batch_size_dropdown.value),
    }
    with open("config.yaml", "w") as _f:
        yaml.dump(_config, _f, default_flow_style=False)
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Run Training Script", kind="success")
    return (train_button,)


@app.cell
def _(logged_in, mo, re, subprocess, train_button):
    training_output = None

    if train_button.value:
        if not logged_in:
            training_output = mo.callout(
                mo.md("**Please log in to W&B first** (W&B Login tab) before training."),
                kind="warn",
            )
        else:
            # Run the training script as a subprocess
            result = subprocess.run(
                ["uv", "run", "python", "train.py"],
                capture_output=True,
                text=True,
                cwd=".",
            )

            console_output = result.stdout + result.stderr
            
            # Extract W&B run link from output
            wandb_link = None
            link_match = re.search(r'https://wandb\.ai/[^\s]+', console_output)
            if link_match:
                wandb_link = link_match.group(0)
            
            # Scrollable console output with max height
            console_html = mo.Html(
                f'<pre style="max-height: 300px; overflow-y: auto; padding: 12px; '
                f'background: #1e1e1e; color: #d4d4d4; border-radius: 6px; '
                f'font-size: 12px; line-height: 1.4;">{console_output}</pre>'
            )

            if result.returncode == 0:
                if wandb_link:
                    success_msg = f"**Training Complete!** 📊 [View your W&B run: {wandb_link}]({wandb_link})"
                else:
                    success_msg = "**Training Complete!**"
                
                training_output = mo.vstack([
                    mo.callout(mo.md(success_msg), kind="success"),
                    mo.accordion({"(Optional) Training script output": console_html}),
                ])
            else:
                training_output = mo.vstack([
                    mo.callout(
                        mo.md("**Training failed.** See the error output below."),
                        kind="danger",
                    ),
                    mo.accordion({"Console output": console_html}),
                ])

    return (training_output,)


@app.cell
def _(batch_size_dropdown, epochs_slider, lr_slider, mo, train_button, training_output):
    training_content = mo.vstack([
        mo.md(
            """
            ##

            **Training** is the process of teaching the neural network to recognize patterns. 
            In our case, the model will look at thousands of handwritten digit images, compare its predictions 
            to the correct answers, and gradually adjust its internal weights to improve accuracy.
            """
        ),
        mo.accordion({
            "(Optional) How a neural network works": mo.md(
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
            ),
        }),
        mo.md(
            """
            ##

            Let's configure your training run. **Hyperparameters** are settings you choose *before* 
            training begins — unlike the model's internal weights (which are learned automatically), 
            these are decisions you make that control *how* the model learns. 
            
            Adjust the three settings below and observe how they affect the model's accuracy in W&B.
            """
        ),
        mo.hstack([
            mo.vstack([
                mo.md("**Epochs**"),
                mo.hstack([epochs_slider, mo.md(f"**{epochs_slider.value}**")], justify="start", gap=1),
                mo.md(
                    """
                    One **epoch** is one complete pass through all 60,000 training images. During each 
                    epoch, the model sees every example once and adjusts its weights to reduce errors.

                    **Too few epochs (1-2):** The model hasn't seen the data enough times to learn 
                    the patterns. This is called **underfitting** — the model is too simple and 
                    performs poorly on both training and test data.

                    **Too many epochs (20+):** The model starts memorizing the training data instead 
                    of learning general patterns. This is called **overfitting** — training accuracy 
                    keeps improving but test accuracy plateaus or gets worse.

                    **Signs of overfitting:** Watch W&B charts — if training accuracy keeps rising 
                    but test accuracy stops improving, you've trained too long.

                    **Suggestion:** Start with 5 epochs. If test accuracy is still improving at the 
                    end, try 10. If it plateaus early, you might need fewer.
                    """
                ),
            ], align="start"),
            mo.vstack([
                mo.md("**Learning Rate**"),
                mo.hstack([lr_slider, mo.md(f"**{lr_slider.value:.4f}**")], justify="start", gap=1),
                mo.md(
                    """
                    The **learning rate** controls how much the model adjusts its weights after each 
                    batch. Think of it as the "step size" when walking downhill to find the lowest 
                    point (the best model).

                    **Too high (0.01-0.1):** The model takes huge steps and might overshoot the 
                    optimal solution. You'll see the loss jumping around erratically or even 
                    increasing instead of decreasing. The model never settles into a good state.

                    **Too low (0.0001):** The model takes tiny, cautious steps. Learning is very 
                    slow — you might need 10x more epochs to reach the same accuracy. The model 
                    might also get stuck in a suboptimal solution.

                    **Just right (0.001):** A balanced step size that allows steady improvement 
                    without instability. This is a common default for the Adam optimizer.

                    **Suggestion:** The default 0.001 works well for most cases. Try 0.01 to see 
                    what unstable training looks like — it's educational!
                    """
                ),
            ], align="start"),
            mo.vstack([
                mo.md("**Batch Size**"),
                batch_size_dropdown,
                mo.md(
                    """
                    Instead of updating weights after every single image, we process images in 
                    **batches**. The batch size is how many images we look at before making one 
                    weight update.

                    **Smaller batches (32):** More frequent weight updates with "noisier" gradients. 
                    This noise can actually help escape bad local minima. Uses less GPU memory. 
                    Each epoch takes longer (more updates to compute).

                    **Larger batches (256):** Fewer updates per epoch with smoother, more stable 
                    gradients. Training is faster per epoch. However, the model may converge to 
                    less optimal solutions. Uses more memory.

                    **The math:** With 60,000 training images and batch_size=64, you get 
                    60,000 ÷ 64 = 938 weight updates per epoch.

                    **Suggestion:** 64 is a solid default. Try 32 if you want potentially better 
                    generalization, or 256 if you want faster training.
                    """
                ),
            ], align="start"),
        ], justify="start", gap=3, align="start"),
        mo.md(
            """
            ##

            Now that you've configured your settings, click the button below to start training. 
            This will download the MNIST dataset (if not cached), train for the configured number 
            of epochs, log metrics to W&B, and evaluate on the test set.
            """
        ),
        train_button,
        training_output,
        mo.md(
            """
            ##

            Once training completes, click the W&B run link in the console output to view your 
            experiment dashboard. Here's what each metric means and what to look for:
                        
            | Metric | Description | What to Look For |
            |:-------|:------------|:-----------------|
            | **batch_loss** | The loss (error) on each batch of images, logged every 100 batches | Should trend downward with some noise. High variance is normal. If jumping wildly or increasing, learning rate is too high. |
            | **epoch_loss** | Average loss across all batches in one epoch — a smoother view | Should decrease steadily each epoch. Plateaus mean the model has learned what it can. Increases indicate a problem. |
            | **epoch_accuracy** | Percentage of training images classified correctly per epoch | Should increase over time. MNIST typically reaches 95-99%. If this rises but test_accuracy stops, you're overfitting. |
            | **test_accuracy** | Final accuracy on 10,000 unseen test images — your "report card" | The metric that matters most. Good MNIST models reach 97-98%. Compare across runs to find the best hyperparameters. |


            W&B charts use "step" as the default x-axis. Each call to 
            `wandb.log()` increments the step counter by 1. In our training script, we log `batch_loss` 
            every 100 batches (~9 times per epoch) and `epoch_loss`/`epoch_accuracy` once per epoch. 
            This means steps don't map directly to epochs — you can change the x-axis to "epoch" in 
            W&B by clicking the chart settings (gear icon) if you prefer that view.

            **Also in W&B — Tables:**
            
            | Table | What It Shows |
            |:------|:--------------|
            | **sample_test_predictions** | 100 random test images with true vs predicted labels. Great for intuition about model performance. |
            | **misclassified_test_examples** | All images where the model was wrong. Study these to understand weaknesses — often ambiguous handwriting even humans struggle with! |
            """
        ),
    ])
    return (training_content,)


# ============================================================================
# TAB 5: Test Model - UI Elements
# ============================================================================

@app.cell
def _(mo):
    file_upload = mo.ui.file(
        filetypes=[".png", ".jpg", ".jpeg", ".gif", ".bmp"],
        label="Upload a digit image",
    )
    return (file_upload,)


@app.cell
def _(Image, SimpleNN, file_upload, io, mo, np, torch, transforms):
    prediction_result = None

    if file_upload.value and len(file_upload.value) > 0:
        uploaded_file = file_upload.value[0]
        image_bytes = uploaded_file.contents

        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = img.convert("L")
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32)

            if img_array.mean() > 127:
                img_array = 255 - img_array

            img_array = img_array / 255.0
            img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
            normalize = transforms.Normalize((0.1307,), (0.3081,))
            img_tensor = normalize(img_tensor)

            try:
                model_to_use = SimpleNN()
                model_to_use.load_state_dict(torch.load("mnist_model.pth", weights_only=True))
                model_to_use.eval()
            except FileNotFoundError:
                prediction_result = mo.callout(
                    mo.md("**No trained model found.** Please train the model first (Step 3 tab)."),
                    kind="warn",
                )
                model_to_use = None

            if model_to_use is not None:
                with torch.no_grad():
                    _outputs = model_to_use(img_tensor)
                    probabilities = torch.softmax(_outputs, dim=1).squeeze()
                    predicted_digit = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_digit].item() * 100

                confidence_data = [
                    {"Digit": i, "Confidence": f"{probabilities[i].item() * 100:.2f}%"}
                    for i in range(10)
                ]

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
                    mo.md("Here's how confident the model is for each digit:"),
                    mo.ui.table(confidence_data),
                ])

        except Exception as e:
            prediction_result = mo.callout(
                mo.md(f"**Error processing image:** {str(e)}"),
                kind="danger",
            )
    else:
        prediction_result = mo.md("_Upload an image above to test the model._")

    return (prediction_result,)


@app.cell
def _(file_upload, mo, prediction_result):
    test_content = mo.vstack([
        mo.md(
            """
            Now let's see your model in action! Upload an image of a handwritten digit to test 
            what it predicts.

            **Tips for best results:** Use a clear image of a single digit (0-9). White digit on 
            dark background works best (like MNIST). The image will be automatically resized to 28x28 pixels.
            """
        ),
        file_upload,
        prediction_result,
    ])
    return (test_content,)


# ============================================================================
# MAIN: Assemble Tabs
# ============================================================================

@app.cell
def _(intro_content, login_content, mo, test_content, training_content):
    mo.vstack([
        mo.md("# Training a neural network with W&B Models"),
        mo.ui.tabs({
            f"{mo.icon('lucide:home')} Introduction": intro_content,
            f"{mo.icon('lucide:settings')} 1. Setup": login_content,
            f"{mo.icon('lucide:bot')} 2: Training": training_content,
            f"{mo.icon('lucide:eye')} 3: Test Model": test_content,
        })
    ])
    return


if __name__ == "__main__":
    app.run()
