from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

import numpy as np
import pickle
import cv2

from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN


# Load stored embeddings and filenames
feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))


# Load VGGFace model
model = VGGFace(
    model='resnet50',
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg'
)


# Create face detector
detector = MTCNN()


# Load query image
sample_img = cv2.imread('sample/ronit.png')

# Convert BGR to RGB
sample_img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)


# Detect face
results = detector.detect_faces(sample_img_rgb)

if len(results) == 0:
    print("No face detected.")
    exit()


# Get first detected face
x, y, width, height = results[0]['box']

x = max(0, x)
y = max(0, y)

face = sample_img_rgb[y:y+height, x:x+width]


# Resize face
face = cv2.resize(face, (224, 224))

# Convert to float32
face = face.astype('float32')


# Add batch dimension
face = np.expand_dims(face, axis=0)


# Preprocess image
face = preprocess_input(face)


# Generate face embedding
result = model.predict(face).flatten()


# Compare with stored embeddings
similarity = []

for feature in feature_list:

    score = cosine_similarity(
        result.reshape(1, -1),
        feature.reshape(1, -1)
    )[0][0]

    similarity.append(score)


# Find best match
index_pos = np.argmax(similarity)

print("Similarity:", similarity[index_pos])
print("Matched file:", filenames[index_pos])


# Display matched image
matched_image = cv2.imread(filenames[index_pos])

cv2.imshow("Matched Celebrity", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
