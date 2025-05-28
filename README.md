# Simple MNIST Digit Recognizer

This project demonstrates a basic neural network for recognizing handwritten digits from the MNIST dataset. It logs training and evaluation metrics to Weights & Biases, providing a hands-on way to understand the fundamentals of AI model training.

## How it Works

### The Dataset: MNIST

*   **What it is**: The script uses the **MNIST dataset**, a collection of 70,000 grayscale images of handwritten digits (0-9).
    *   60,000 images are used for training the model.
    *   10,000 images are used for testing the model's performance on unseen data.
*   **Image Format**: Each image is 28x28 pixels. These pixel values are normalized (scaled) to help with the training process.
*   **Purpose**: MNIST is a classic dataset for beginners in image classification due to its simplicity and well-defined task.

The dataset is automatically downloaded by the `train.py` script if not found locally in a `./data` directory.

### The Model: A Simple Neural Network (`SimpleNN`)

The model is a feed-forward neural network designed to classify the digit in an input image.

1.  **Input**: The model takes a flattened 28x28 pixel image (784 numerical values) as input.
2.  **Architecture**:
    *   **Layer 1 (Fully Connected)**: `nn.Linear(28 * 28, 128)` - This layer transforms the 784 input pixel values into 128 intermediate features. It starts to identify basic patterns.
    *   **Activation (ReLU)**: `nn.ReLU()` - This function introduces non-linearity, allowing the model to learn more complex relationships by setting negative values to zero.
    *   **Layer 2 (Fully Connected)**: `nn.Linear(128, 10)` - This layer takes the 128 features and transforms them into 10 output values.
3.  **Output**: The 10 output values represent the model's confidence scores for each digit (0 through 9). The digit with the highest score is the model's prediction.

### What the Model is "Learning"

The model learns to **recognize the visual patterns and features** that distinguish one handwritten digit from another.

This learning process happens through training:

1.  **Initialization**: The model's internal parameters (weights and biases) start with random values.
2.  **Forward Pass**: An image is fed to the model, and it makes a prediction.
3.  **Loss Calculation**: The prediction is compared to the true label of the image. A "loss function" (`nn.CrossEntropyLoss`) quantifies how wrong the prediction was.
4.  **Backward Pass (Backpropagation)**: The model calculates how much each internal parameter contributed to the error.
5.  **Optimization**: An "optimizer" (`optim.Adam`) adjusts the model's parameters slightly to reduce the error for future predictions.
6.  **Iteration**: This cycle repeats for many images over multiple passes through the training dataset (epochs).

By continuously adjusting its parameters, the model gets better at mapping the input pixel patterns to the correct digit labels.

## Running the Script

1.  **Install Dependencies**:
    Make sure you have `PyYAML` included in your `requirements.txt` and install it:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure Hyperparameters**:
    Edit the `config.yaml` file to set your desired hyperparameters (e.g., `epochs`, `lr`, `batch_size`).
3.  **Log in to Weights & Biases**:
    If you haven't used W&B before, sign up at [https://wandb.ai/site](https://wandb.ai/site) and then log in via your terminal:
    ```bash
    wandb login
    ```
    You will be prompted for your API key.
4.  **Run the Training Script**:
    ```bash
    python train.py
    ```

## Viewing Metrics and Artifacts

After the script runs, you will see a link in your terminal leading to your Weights & Biases dashboard. There, you can visualize metrics, inspect model predictions, and retrieve saved model artifacts.

### Key Metrics:

*   `batch_loss`: The loss calculated for each batch of images during training.
*   `epoch_loss`: The average loss over an entire epoch (one pass through the training data).
*   `epoch_accuracy`: The percentage of training images correctly classified during an epoch.
*   `test_accuracy`: The final accuracy of the model on the unseen test dataset.

### Model Predictions (W&B Tables):

*   **`sample_test_predictions`**: This table shows a sample of ~100 predictions from the test set. It includes the input image, the true label, and the model's predicted label. This gives a qualitative feel for the model's performance.
*   **`misclassified_test_examples`**: This table specifically logs all the examples from the test set where the model's prediction was incorrect. It displays the input image, true label, and the wrong predicted label. Reviewing these examples is very useful for understanding the types of mistakes the model is making and can guide further improvements.

### Model Artifact:

*   The script also logs the trained model as a W&B Artifact named `mnist-simple-nn` (type: `model`). You can find this in the "Artifacts" section of your W&B run. This allows you to version your models and easily retrieve them for later use (e.g., for inference or to continue training).

## Experimenting with Hyperparameters

Hyperparameters for this project are managed in the `config.yaml` file. This file dictates the settings the model uses for training, such as the number of epochs, learning rate, and batch size.

```yaml
# Example config.yaml
epochs: 5
lr: 0.001
batch_size: 64
```

Experimenting with these is a core part of machine learning.

### How to Experiment Systematically:

1.  **Change One Hyperparameter at a Time:** This is crucial. If you modify multiple settings at once, it's difficult to determine which change caused an observed effect.
2.  **Edit `config.yaml`:** Directly modify the values in the `config.yaml` file.
    *Example: To test a learning rate of `0.01` for `10` epochs:* 
    ```yaml
    epochs: 10
    lr: 0.01
    batch_size: 64 # Assuming you keep batch_size the same for this experiment
    ```
3.  **Use Descriptive Run Names in W&B (Optional but Recommended):** While the config will be logged automatically, you might still want to give your W&B runs specific names to easily identify them. You can do this by modifying the `wandb.init()` line in `train.py` temporarily for a specific set of experiments, or by renaming runs in the W&B UI.
    ```python
    # In train.py, for a specific experimental run:
    wandb.init(project="simple-mnist-training", config=yaml_config, name="experiment_lr_0.01_epochs_10")
    ```
    Remember to remove or adjust the `name` parameter for subsequent standard runs if you don't want to manually name each one.
4.  **Observe the Metrics in W&B:** After running `python train.py` with your modified `config.yaml`, pay close attention to the W&B dashboard:
    *   `epoch_loss` & `batch_loss`: How quickly and stably does loss decrease?
    *   `epoch_accuracy` (training accuracy): How well does the model learn the training data?
    *   `test_accuracy`: How well does the model generalize to new, unseen data? This is often the most important metric.
    *   The `misclassified_test_examples` table: Do different settings lead to different kinds of mistakes?

### Key Hyperparameters to Explore (in `config.yaml`):

#### 1. Learning Rate (`lr`)
*   **What it is:** Controls how much the model's weights are adjusted during each optimization step. Think of it as the step size.
*   **Default Value in `config.yaml`:** `0.001`
*   **To Try in `config.yaml`:**
    *   **Increase (e.g., `0.01`, `0.1`):**
        *   *Potential Upside:* Learns faster initially.
        *   *Potential Downside:* If too high, loss might fluctuate wildly, jump around, or even increase (model "overshoots"). Poor accuracy.
    *   **Decrease (e.g., `0.0001`):**
        *   *Potential Upside:* More precise adjustments, potentially better final loss/accuracy.
        *   *Potential Downside:* Training will be much slower. Might take many more epochs or get "stuck."

#### 2. Number of Epochs (`epochs`)
*   **What it is:** One complete pass through the entire training dataset.
*   **Default Value in `config.yaml`:** `5`
*   **To Try in `config.yaml`:**
    *   **Decrease (e.g., `2`):**
        *   *Potential Effect:* Model won't see data enough, likely leading to **underfitting** (lower training and test accuracy).
    *   **Increase (e.g., `10`, `20`):**
        *   *Potential Upside:* Model learns more, potentially improving accuracy.
        *   *Potential Downside (Overfitting):* Model might memorize training data instead of learning general patterns. Look for: training accuracy continuing to improve while test accuracy plateaus or decreases.

#### 3. Batch Size (`batch_size`)
*   **What it is:** Number of training examples used in one iteration before weights are updated.
*   **Default Value in `config.yaml`:** `64`
*   **To Try in `config.yaml`:**
    *   **Decrease (e.g., `16`, `32`):**
        *   *Potential Effects:* More frequent but "noisier" updates. Can sometimes help escape poor local minima. Slower epochs. Might need a smaller learning rate. Uses less memory.
    *   **Increase (e.g., `128`, `256`):**
        *   *Potential Effects:* Less frequent, "smoother" updates. Faster epochs (if hardware supports it). Can sometimes converge to less optimal solutions. Uses more memory (risk of out-of-memory errors).

### Your First Experiments:

1.  Open `config.yaml`.
2.  Start by changing only the **learning rate (`lr`)**. Try `0.01`.
3.  Run `python train.py`. Give your W&B run a descriptive name if you wish (either in the UI or by temporarily editing `train.py`).
4.  Change `lr` in `config.yaml` to `0.0001`. Run `python train.py` again.
5.  Compare these two runs and the original run (with `lr: 0.001`) in Weights & Biases. Observe differences in loss and accuracy curves.
6.  Next, reset `lr` to `0.001` in `config.yaml` and try changing `epochs`.

This iterative process is fundamental to applied machine learning. Happy experimenting!

## Serving the Model with an API Endpoint

This project includes a FastAPI server (`serve.py`) to expose the trained MNIST model as an API endpoint. This allows you to send image data to the model and receive predictions programmatically.

### Run the API Server

To start the API server, navigate to your project directory in the terminal and run:

```bash
uvicorn serve:app
```

You should see output indicating the server is running, typically on `http://127.0.0.1:8000`.

## Interactive Web UI for Predictions

This project also includes a simple web interface to upload images and get predictions directly in your browser. Once the server is running, open your web browser and navigate to:

[http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### Using the Interface

1.  **Choose File**: Click the "Choose File" button to select an image of a digit from your computer (PNG, JPG, GIF are generally supported by browsers).
2.  **Image Preview**: A small preview of your selected image will be shown.
3.  **Predict**: Click the "Predict" button.
4.  **Loading**: A "Predicting..." message will appear while the image is processed and sent to the API.
5.  **Results**: The predicted digit will be displayed prominently, along with a list of confidence scores for all 10 digits (0-9).

**Note on Image Processing:** The UI performs basic client-side processing to resize the image to 28x28 pixels and convert it to grayscale before sending the pixel data to the `/predict` API endpoint. The quality of the prediction can be sensitive to how well the uploaded image resembles the MNIST dataset's characteristics (e.g., a relatively centered digit, contrast between digit and background). 