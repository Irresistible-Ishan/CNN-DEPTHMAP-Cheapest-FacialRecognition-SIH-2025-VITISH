# this file creates a new embedding everytime a new person 's face is added
# then it matches the new image manually written and given
# tis jsut a test that CNN works

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable # ✅ 1. IMPORT THIS
from PIL import Image
import os


IMG_SIZE = (160, 160)
EMBEDDING_FILE = "embeddings_withnewpersonadded1.npz"  # new embedding file

@register_keras_serializable() 
def l2_normalize(x):
    return tf.math.l2_normalize(x, axis=1)

embedding_model = tf.keras.models.load_model("face_embedding_model.keras")


# Load existing embeddings 
if os.path.exists(EMBEDDING_FILE):
    data = np.load(EMBEDDING_FILE)
    embeddings = data['embeddings']
    labels = data['labels']
else:
    embeddings = np.empty((0, 128)) 
    labels = np.empty((0,))


#Preprocess a single image compatible for model 
# later on will do this step automatically 
def preprocess_image(filename):
    img = Image.open(filename).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)  # (1, H, W, C)



# to Add new face
def add_new_face(filename, person_id):
    global embeddings, labels
    img = preprocess_image(filename)
    emb = embedding_model.predict(img)
    embeddings = np.vstack([embeddings, emb])
    labels = np.hstack([labels, person_id])
    np.savez(EMBEDDING_FILE, embeddings=embeddings, labels=labels)
    print(f"✅ New face added for ID {person_id} and saved to {EMBEDDING_FILE}")


# Search face using search method
def search_face(filename, threshold=0.9):
    if len(embeddings) == 0:
        print("❌ No embeddings in database yet.")
        return
    
    img = preprocess_image(filename)
    emb = embedding_model.predict(img)  # shape (1, 128)
    
    # Compute cosine similarity
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    sims = np.dot(embeddings, emb.T).flatten() / norms
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    
    if best_score >= threshold:
        print(f"✅ Match found! Label: {labels[best_idx]}, Similarity: {best_score:.3f}")
    else:
        print(f"❌ No match found. Highest similarity: {best_score:.3f}")


# Menu Main test code

def menu():
    while True:
        print("\n--- Face Embedding Menu ---")
        print("1 - Add new face")
        print("2 - Search face")
        print("3 - Exit")
        choice = input("Enter choice: ").strip()
        
        if choice == "1":
            file = input("Enter image file path: ").strip()
            pid = int(input("Enter person ID: ").strip())
            add_new_face(file, pid)
        elif choice == "2":
            file = input("Enter image file path to search: ").strip()
            search_face(file)
        elif choice == "3":
            print("Exiting.")
            break
        else:
            print("❌ Invalid choice.")

# --------------------------
# Run menu
# --------------------------
if __name__ == "__main__":
    menu()
