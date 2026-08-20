from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity

import cv2
from mtcnn import MTCNN
from PIL import Image


# ---------------------------------------------------
# 1. Load stored embeddings and filenames
# ---------------------------------------------------

feature_list = np.array(
    pickle.load(open('embedding.pkl', 'rb'))
)

filenames = pickle.load(
    open('filenames.pkl', 'rb')
)

print("Number of stored embeddings:", len(feature_list))
print("Number of filenames:", len(filenames))


# ---------------------------------------------------
# 2. Load VGGFace model
# ---------------------------------------------------

model = VGGFace(
    model='resnet50',
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg'
)


# ---------------------------------------------------
# 3. Create MTCNN detector
# ---------------------------------------------------

detector = MTCNN()


# ---------------------------------------------------
# 4. Load query image
# ---------------------------------------------------

sample_img = cv2.imread('sample/ronit.png')

if sample_img is None:
    print("Image could not be loaded.")
    exit()


# ---------------------------------------------------
# 5. Convert BGR -> RGB
# ---------------------------------------------------

sample_img_rgb = cv2.cvtColor(
    sample_img,
    cv2.COLOR_BGR2RGB
)


# ---------------------------------------------------
# 6. Detect face
# ---------------------------------------------------

results = detector.detect_faces(sample_img_rgb)

print("Number of faces detected:", len(results))

if len(results) == 0:
    print("No face detected.")
    exit()


# ---------------------------------------------------
# 7. Get the first detected face
# ---------------------------------------------------

x, y, width, height = results[0]['box']

# MTCNN can sometimes return negative x/y values
x = max(0, x)
y = max(0, y)

# Make sure width and height are valid
width = max(1, width)
height = max(1, height)


# ---------------------------------------------------
# 8. Crop face
# ---------------------------------------------------

face = sample_img_rgb[
    y:y + height,
    x:x + width
]


# ---------------------------------------------------
# 9. Show detected face
# ---------------------------------------------------

cv2.imshow(
    'Detected Face',
    cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ---------------------------------------------------
# 10. Resize face to 224x224
# ---------------------------------------------------

image = Image.fromarray(face)

image = image.resize((224, 224))


# ---------------------------------------------------
# 11. Convert image to NumPy array
# ---------------------------------------------------

face_array = np.asarray(image)

face_array = face_array.astype('float32')


print("Face array shape:", face_array.shape)


# ---------------------------------------------------
# 12. Add batch dimension
# ---------------------------------------------------

expanded_img = np.expand_dims(
    face_array,
    axis=0
)

print("Expanded image shape:", expanded_img.shape)


# ---------------------------------------------------
# 13. VGGFace preprocessing
# ---------------------------------------------------

preprocessed_img = preprocess_input(
    expanded_img
)


# ---------------------------------------------------
# 14. Generate embedding
# ---------------------------------------------------

result = model.predict(
    preprocessed_img
).flatten()

print("Query embedding shape:", result.shape)


# ---------------------------------------------------
# 15. Compare with stored embeddings
# ---------------------------------------------------

similarity = []

for i in range(len(feature_list)):

    score = cosine_similarity(
        result.reshape(1, -1),
        feature_list[i].reshape(1, -1)
    )[0][0]

    similarity.append(score)


# ---------------------------------------------------
# 16. Find highest similarity
# ---------------------------------------------------

index_pos = np.argmax(similarity)

best_similarity = similarity[index_pos]

print("Best similarity:", best_similarity)

print("Matched file:", filenames[index_pos])


# ---------------------------------------------------
# 17. Load matched image
# ---------------------------------------------------

temp_img = cv2.imread(
    filenames[index_pos]
)


# ---------------------------------------------------
# 18. Show result
# ---------------------------------------------------

cv2.imshow(
    'Output',
    temp_img
)

cv2.waitKey(0)
cv2.destroyAllWindows()