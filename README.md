# 🌿 Crop Disease Detection

A computer-vision application that identifies crop diseases from leaf images using a fine-tuned MobileNet model. The project includes a TensorFlow/Keras training notebook and a Streamlit web interface for uploading plant images and viewing a disease prediction with confidence.

## Project Overview

This repository was developed to classify crop leaves from the **New Plant Diseases Dataset**. The model recognizes 38 plant-disease classes and returns the most likely class along with its confidence score.

### Features

- Upload JPEG, PNG, or JPG leaf images.
- Predict the crop disease class using the saved Keras model.
- Display prediction confidence and a progress indicator.
- Show a warning for low-confidence or unrelated images.
- Include the complete training workflow in a Jupyter notebook.
- Use a 224 × 224 image input for prediction.

### Supported Model

The application loads `model_4_mobilenet_finetuned.keras`, which contains the trained MobileNet-based classifier. The model is expected to be kept in the project root because the Streamlit application resolves its model path relative to `app.py`.

## Dataset

The project uses the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle.

The training notebook expects the dataset to be available through Kaggle's input directory in the following structure:

```text
/kaggle/input/new-plant-diseases-dataset/
└── New Plant Diseases Dataset(Augmented)/
    └── New Plant Diseases Dataset(Augmented)/
        ├── train/
        └── valid/
```

The notebook uses the training directory to create the training and validation datasets and the validation directory for testing.

> Note: The application does not need the dataset at runtime. It only requires the trained `.keras` model, which is already included in this repository.

## Tech Stack

- **Frontend:** Streamlit
- **Deep learning:** TensorFlow and Keras
- **Data processing:** NumPy and Pillow
- **Image format:** JPG, JPEG, and PNG
- **Model:** MobileNet fine-tuned classifier
- **Dataset:** New Plant Diseases Dataset

## Prerequisites

- Python 3.10 or later
- Git
- A Kaggle account and API token if you plan to retrieve the dataset programmatically
- A CPU or GPU-compatible TensorFlow installation

The repository includes a TensorFlow CPU dependency, which is suitable for local CPU-based inference and training.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Ryougi-Shikii/CropDiseaseDetection.git
   cd CropDiseaseDetection
   ```

2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   On macOS or Linux, use:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the project dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Install Jupyter Notebook support if you want to run the training notebook:

   ```bash
   python -m pip install notebook
   ```

## Run the Application

From the project root, start the Streamlit application:

```bash
streamlit run app.py
```

Open the URL displayed by Streamlit, usually:

```text
http://localhost:8501
```

The application will automatically load the pretrained model. Select an image, click **Analyze Leaf**, and review the predicted disease and confidence score.

## Training the Model

The complete training workflow is in `Complete Model Training.ipynb`.

### Training prerequisites

- Access to the New Plant Diseases Dataset on Kaggle.
- A Kaggle notebook or local environment with access to Kaggle datasets.
- The Kaggle dataset path used by the notebook.
- TensorFlow and optional Jupyter Notebook dependencies.

### Dataset preparation

The notebook loads:

- Training data from the dataset's `train` directory.
- Validation data using an internal 20% split from the training directory.
- Test data from the dataset's `valid` directory.

If you are running locally, place the extracted dataset in a location consistent with the notebook's configured path, or update `DATASET_PATH` and `TEST_DIR_PATH` in the notebook to match your local directory layout.

### Training commands

Open the notebook and run its cells in order:

```bash
jupyter notebook
```

Select `Complete Model Training.ipynb`, then run all cells. The notebook trains several model variants and saves the final model as:

```text
model_4_mobilenet_finetuned.keras
```

The model file is already included in the repository. To retrain and replace it, run the notebook's final model-saving cell and make sure the resulting file is copied to the project root.

## Model Classes

The application contains 38 class labels. These labels include healthy and diseased examples for crops such as apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato, raspberry, soybean, squash, strawberry, and tomato.

The complete class list is available in the Streamlit sidebar when the application starts.

## Project Structure

```text
CropDiseaseDetection/
├── app.py
├── Complete Model Training.ipynb
├── model_4_mobilenet_finetuned.keras
├── requirements.txt
└── README.md
```

## Usage Notes

- The model accepts images containing a visible crop leaf.
- Clear, centered, and well-lit leaves provide the best results.
- The interface uses a 70% confidence threshold to distinguish a recognized class from an unknown image.
- Predictions are labels from the training dataset and should not be interpreted as a medical or regulatory diagnosis.
- The application uses the model's saved class order; class names should remain aligned with the list in `app.py`.

## Troubleshooting

### Model file not found

Ensure `model_4_mobilenet_finetuned.keras` is in the same directory as `app.py`. If the model was retrained, copy the saved artifact into the project root.

### TensorFlow import or GPU error

Verify that the active environment uses the expected Python interpreter and that TensorFlow is installed:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

A GPU is optional for inference; CPU execution is supported by the included `tensorflow-cpu` dependency.

### Streamlit cannot start

Validate the installation and run the app from the repository root:

```bash
python -m pip check
streamlit run app.py
```

### Dataset loading errors

The training notebook expects Kaggle's dataset directory. Confirm that the Kaggle dataset is accessible and update the notebook paths if your local dataset layout differs.

## License

See the repository's license file if one is included in the project history or branch. The dataset's Kaggle terms and license must also be reviewed before redistributing or using its contents.
