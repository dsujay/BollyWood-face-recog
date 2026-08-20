[README(1).md](https://github.com/user-attachments/files/31270057/README.1.md)
# Bollywood Celebrity Face Recognition

A face-recognition project that identifies a Bollywood celebrity from an input image using a **pretrained VGGFace model with a ResNet50 backbone**. The system generates a numerical face embedding for the input image and compares it with stored celebrity-image embeddings using **cosine similarity**.

## Project Overview

The project contains a dataset of images organized into celebrity folders. The current dataset contains around **100 celebrities** and thousands of images.

The main pipeline is:

```text
Celebrity Dataset
      ↓
Collect image paths
      ↓
VGGFace + ResNet50
      ↓
Generate face embeddings
      ↓
Save embeddings
      ↓
Input / Query Image
      ↓
MTCNN Face Detection
      ↓
Face Cropping
      ↓
VGGFace + ResNet50
      ↓
Query Embedding
      ↓
Cosine Similarity
      ↓
Best Matching Celebrity Image
```

## Technologies Used

- Python
- TensorFlow
- Keras
- Keras-VGGFace
- ResNet50
- NumPy
- OpenCV
- MTCNN
- Pillow (PIL)
- scikit-learn
- Pickle
- tqdm

## Project Structure

```text
project1_bollywood/
│
├── data/
│   ├── Celebrity_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Celebrity_2/
│   └── ...
│
├── sample/
│   └── query_image.jpg
│
├── feature_extractor.py
├── test.py
├── filenames.pkl
├── embedding.pkl
└── README.md
```

## How It Works

### 1. Collect image paths

The dataset contains one folder for each celebrity. `feature_extractor.py` traverses all celebrity folders and stores the paths of all images in `filenames.pkl`.

Example:

```text
data/AamirKhan/image1.jpg
data/AamirKhan/image2.jpg
data/SalmanKhan/image1.jpg
...
```

### 2. Generate face embeddings

The VGGFace model is loaded with a ResNet50 backbone:

```python
model = VGGFace(
    model='resnet50',
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg'
)
```

`include_top=False` removes the original classification head, allowing the model to be used as a feature extractor.

`pooling='avg'` converts the final feature maps into a **2048-dimensional vector**.

Each dataset image is converted into an embedding and the embeddings are saved in:

```text
embedding.pkl
```

### 3. Detect the face in a query image

`test.py` uses **MTCNN** to detect a face and obtain its bounding box.

The detected face is cropped and resized to:

```text
224 × 224 × 3
```

### 4. Create the query embedding

The cropped face is preprocessed and passed through the same VGGFace/ResNet50 feature extractor to create a 2048-dimensional query embedding.

### 5. Compare embeddings

The query embedding is compared with all stored embeddings using **cosine similarity**.

The embedding with the highest similarity score is selected as the best match.

## Running the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd project1_bollywood
```

### 2. Create and activate a virtual environment

This project uses an older TensorFlow/VGGFace stack. The working environment used during development was based on Python 3.8.

```powershell
py -3.8 -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the packages required by the project. For the legacy TensorFlow/VGGFace setup, use versions compatible with the project environment rather than blindly installing the latest releases.

Example packages used during development:

```powershell
python -m pip install tensorflow==2.3.1
python -m pip install protobuf==3.20.3
python -m pip install numpy==1.19.3
python -m pip install keras==2.4.3
python -m pip install mtcnn==0.1.0
python -m pip install tqdm
python -m pip install scikit-learn
python -m pip install opencv-python
python -m pip install pillow
```

`keras-vggface` is also required:

```powershell
python -m pip install keras-vggface
```

> Because this project uses legacy packages, package compatibility is important. Avoid upgrading TensorFlow, Keras, NumPy, or protobuf indiscriminately after creating the environment.

### 4. Generate filenames

Run the script that collects all dataset image paths and creates:

```text
filenames.pkl
```

### 5. Generate embeddings

Run:

```powershell
python feature_extractor.py
```

This processes the dataset images and creates:

```text
embedding.pkl
```

### 6. Run face recognition

Put a query image inside the `sample` folder and update the path in `test.py` if required.

Then run:

```powershell
python test.py
```

The program detects the face, generates its embedding, compares it with the stored embeddings, and displays the best matching image.

## Important Model Details

- **Face detector:** MTCNN
- **Feature extractor:** VGGFace
- **Backbone:** ResNet50
- **Input size:** 224 × 224 × 3
- **Embedding size:** 2048
- **Similarity metric:** Cosine similarity
- **Recognition type:** 1-to-N face identification

## Why Cosine Similarity?

The VGGFace model represents each face as a 2048-dimensional vector. Cosine similarity measures how similar two vectors are based on their direction.

```text
Higher similarity → more similar embeddings
Lower similarity  → less similar embeddings
```

The system compares the query embedding with all stored embeddings and chooses the highest score.

## Important Limitations

### 1. Training vs. inference preprocessing

The current implementation generates the stored embeddings from the original dataset images, while the query image is first processed with MTCNN and cropped. Ideally, **the same face detection, cropping, and preprocessing pipeline should be used for both reference images and query images**.

A stronger version of the project would regenerate `embedding.pkl` after applying MTCNN consistently to the reference images.

### 2. No explicit accuracy evaluation yet

The recognition script does not automatically calculate model accuracy. To evaluate the complete system, a separate test set should be kept aside and the percentage of correctly identified images should be measured.

### 3. No unknown-person threshold

The current system always returns the highest-similarity match. A future version should use a similarity threshold so that a person who is not present in the database can be returned as **Unknown** instead of being forced into one of the known celebrities.

### 4. Linear search

The current implementation compares the query embedding with every stored embedding. This is suitable for a small dataset but becomes slower as the number of embeddings grows.

For a larger system, an approximate nearest-neighbor index such as **FAISS** or **HNSW** could be used.

### 5. Multiple faces

The current code uses the first detected face:

```python
results[0]['box']
```

For group photos, a stronger implementation would process every detected face separately.

## Future Improvements

- Use MTCNN consistently during reference embedding generation.
- Add face alignment using MTCNN landmarks.
- Add an **Unknown** class using a similarity threshold.
- Evaluate the system using a separate test set and report top-1/top-k accuracy.
- Replace the brute-force search with FAISS for larger datasets.
- Build a Streamlit web interface for image upload and prediction.
- Consider modern face-recognition models such as FaceNet or ArcFace.

## Example Output

```text
Number of stored embeddings: 8664
Number of filenames: 8664
Number of faces detected: 1
Face array shape: (224, 224, 3)
Expanded image shape: (1, 224, 224, 3)
Query embedding shape: (2048,)
Best similarity: <score>
Matched file: data/<celebrity>/<image>.jpg
```

## Disclaimer

This is an educational face-recognition project intended for learning computer vision, deep learning, embeddings, and similarity search.
