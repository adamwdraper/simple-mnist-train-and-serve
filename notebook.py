import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Simple MNIST Digit Recognizer

        This interactive notebook demonstrates training a neural network to recognize handwritten digits 
        using the MNIST dataset, with experiment tracking via **Weights & Biases**.

        ## What You'll Learn

        - How a simple neural network classifies images
        - How to track experiments with W&B
        - How hyperparameters affect model performance

        ## The Dataset: MNIST

        The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9):
        - **60,000** images for training
        - **10,000** images for testing
        - Each image is **28x28 pixels**

        ## The Model: SimpleNN

        A 2-layer feed-forward neural network:

        1. **Input**: Flattened 28x28 image (784 values)
        2. **Hidden Layer**: 128 neurons with ReLU activation
        3. **Output**: 10 neurons (one per digit class)

        ---

        **New to W&B?** [Sign up here](https://wandb.ai/site) to get your API key.
        """
    )
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
    return (
        Image,
        accuracy_score,
        io,
        nn,
        np,
        optim,
        torch,
        torchvision,
        transforms,
        wandb,
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
    mo.md(
        """
        ## Step 2: Configure Hyperparameters

        Adjust the training settings below. Experiment with different values to see how they affect model performance!
        """
    )
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

    lr_slider = mo.ui.slider(
        start=0.0001,
        stop=0.1,
        value=0.001,
        step=0.0001,
        label="Learning Rate",
        show_value=True,
    )

    batch_size_dropdown = mo.ui.dropdown(
        options={"32": 32, "64": 64, "128": 128, "256": 256},
        value="64",
        label="Batch Size",
    )
    return batch_size_dropdown, epochs_slider, lr_slider


@app.cell
def _(batch_size_dropdown, epochs_slider, lr_slider, mo):
    mo.hstack(
        [
            mo.vstack([mo.md("**Epochs**"), epochs_slider]),
            mo.vstack([mo.md("**Learning Rate**"), lr_slider]),
            mo.vstack([mo.md("**Batch Size**"), batch_size_dropdown]),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell
def _(batch_size_dropdown, epochs_slider, lr_slider, mo):
    config_summary = mo.md(
        f"""
        ### Current Configuration
        | Parameter | Value |
        |-----------|-------|
        | Epochs | {epochs_slider.value} |
        | Learning Rate | {lr_slider.value:.4f} |
        | Batch Size | {batch_size_dropdown.value} |
        """
    )
    config_summary
    return (config_summary,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 3: Train the Model

        Click the button below to start training. The model will:
        1. Download the MNIST dataset (if not cached)
        2. Train for the specified number of epochs
        3. Log metrics to W&B
        4. Evaluate on the test set
        """
    )
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Start Training", kind="success")
    train_button
    return (train_button,)


@app.cell
def _(
    SimpleNN,
    accuracy_score,
    batch_size_dropdown,
    epochs_slider,
    logged_in,
    lr_slider,
    mo,
    nn,
    optim,
    torch,
    torchvision,
    train_button,
    transforms,
    wandb,
):
    training_result = None
    run_url = None
    trained_model = None
    test_accuracy = None

    if train_button.value:
        if not logged_in:
            training_result = mo.callout(
                mo.md("**Please log in to W&B first** (Step 1) before training."),
                kind="warn",
            )
        else:
            mo.output.replace(mo.md("**Initializing training...**"))

            # Device configuration
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Data transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            # Load datasets
            mo.output.replace(mo.md("**Downloading/loading MNIST dataset...**"))
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, transform=transform, download=True
            )
            test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, transform=transform, download=True
            )

            batch_size = int(batch_size_dropdown.value)
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset, batch_size=batch_size, shuffle=False
            )

            # Initialize model
            model = SimpleNN().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr_slider.value)

            # Initialize W&B run (same pattern as train.py)
            run = wandb.init(
                project="simple-mnist-training",
                config={
                    "epochs": epochs_slider.value,
                    "lr": lr_slider.value,
                    "batch_size": batch_size,
                },
            )
            run_url = run.url

            # Training loop
            training_logs = []
            for epoch in range(epochs_slider.value):
                model.train()
                running_loss = 0.0
                epoch_labels = []
                epoch_preds = []

                mo.output.replace(
                    mo.md(f"**Training Epoch {epoch + 1}/{epochs_slider.value}...**")
                )

                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    epoch_preds.extend(predicted.cpu().numpy())
                    epoch_labels.extend(labels.cpu().numpy())

                    if (i + 1) % 100 == 0:
                        wandb.log({"batch_loss": loss.item(), "epoch": epoch, "batch_step": i+1})

                epoch_loss = running_loss / len(train_loader)
                epoch_accuracy = accuracy_score(epoch_labels, epoch_preds)
                training_logs.append({
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "accuracy": epoch_accuracy
                })
                wandb.log({
                    "epoch_loss": epoch_loss,
                    "epoch_accuracy": epoch_accuracy,
                    "epoch": epoch + 1
                })

            # Evaluation (same as train.py)
            mo.output.replace(mo.md("**Evaluating on test set...**"))
            model.eval()
            all_preds = []
            all_labels = []

            # Create W&B Tables
            sample_predictions_table = wandb.Table(columns=["Image", "True Label", "Predicted Label"])
            misclassified_table = wandb.Table(columns=["Image", "True Label", "Predicted Label"])
            MAX_SAMPLES_TO_LOG = 100
            samples_logged = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images_device = images.to(device)
                    outputs = model(images_device)
                    _, predicted = torch.max(outputs.data, 1)

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())

                    # Log samples and misclassifications
                    for j in range(images.size(0)):
                        true_label = labels[j].item()
                        pred_label = predicted[j].item()

                        if samples_logged < MAX_SAMPLES_TO_LOG:
                            sample_img = wandb.Image(images[j])
                            sample_predictions_table.add_data(sample_img, true_label, pred_label)
                            samples_logged += 1

                        if pred_label != true_label:
                            misclassified_img = wandb.Image(images[j])
                            misclassified_table.add_data(misclassified_img, true_label, pred_label)

            test_accuracy = accuracy_score(all_labels, all_preds)
            wandb.log({"test_accuracy": test_accuracy})

            # Log the tables
            wandb.log({"sample_test_predictions": sample_predictions_table})
            wandb.log({"misclassified_test_examples": misclassified_table})

            # Save model locally
            torch.save(model.state_dict(), "mnist_model.pth")

            # Log model artifact
            model_artifact = wandb.Artifact(
                "mnist-simple-nn",
                type="model",
                description="Simple Neural Network trained on MNIST",
                metadata={"epochs": epochs_slider.value, "lr": lr_slider.value, "batch_size": batch_size}
            )
            model_artifact.add_file("mnist_model.pth")
            wandb.log_artifact(model_artifact)

            wandb.finish()

            trained_model = model

            # Build results display
            logs_table = mo.ui.table(
                training_logs,
                label="Training Progress",
            )

            training_result = mo.vstack([
                mo.callout(
                    mo.md(f"**Training Complete!** Test Accuracy: **{test_accuracy * 100:.2f}%**"),
                    kind="success",
                ),
                mo.md(f"**View your run in W&B:** [{run_url}]({run_url})"),
                mo.md("### Training Logs"),
                logs_table,
            ])

            mo.output.replace(training_result)

    training_result if training_result else mo.md(
        "_Click 'Start Training' to begin. Make sure you're logged in to W&B first._"
    )
    return (
        run_url,
        test_accuracy,
        trained_model,
        training_result,
    )


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
    trained_model,
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

            # Get model for inference
            if trained_model is not None:
                model_to_use = trained_model
            else:
                # Try to load saved model
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
