import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import cv2
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
import random

# --- Constants ---
IMG_SIZE = (160, 160)
SIMILARITY_THRESHOLD = 0.8
EMBEDDINGS_DIR = "embeddings_data" # Directory to store all embedding files
CSV_FILE = "student_data.csv"

# --- Keras Custom Object ---
@register_keras_serializable()
def l2_normalize(x):
    return tf.math.l2_normalize(x, axis=1)

# --- Dialog for Adding a New Student ---
class AddStudentDialog(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Add New Student")
        self.geometry("300x200")
        self.transient(master)
        self.grab_set()
        self.result = None

        self.name_label = ctk.CTkLabel(self, text="Name:")
        self.name_label.pack(pady=(10, 0))
        self.name_entry = ctk.CTkEntry(self, placeholder_text="Enter student's name")
        self.name_entry.pack(pady=5, padx=20, fill="x")

        self.roll_label = ctk.CTkLabel(self, text="Roll Number:")
        self.roll_label.pack(pady=(10, 0))
        self.roll_entry = ctk.CTkEntry(self, placeholder_text="Enter roll number")
        self.roll_entry.pack(pady=5, padx=20, fill="x")

        self.submit_button = ctk.CTkButton(self, text="Submit", command=self._on_submit)
        self.submit_button.pack(pady=20)

    def _on_submit(self):
        name = self.name_entry.get().strip()
        rollno = self.roll_entry.get().strip()
        if name and rollno:
            self.result = {"name": name, "rollno": rollno}
            self.destroy()
        else:
            messagebox.showwarning("Input Error", "Please fill in all fields.", parent=self)

    def wait_for_input(self):
        self.wait_window()
        return self.result

# --- Main Application ---
class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Face Recognition Attendance System - SIH 2025")
        self.geometry("800x600")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # --- Initialize Data Holders ---
        self.embedding_model = None
        self.embeddings = np.empty((0, 128))
        self.labels = np.empty((0,), dtype=int)
        self.student_data = pd.DataFrame(columns=["name", "rollno", "attendance_so_far", "idlabel"])

        # --- UI Layout ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.title_label = ctk.CTkLabel(self, text="Face Recognition Attendance", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        self.controls_frame = ctk.CTkFrame(self.main_frame)
        self.controls_frame.grid(row=0, column=0, rowspan=2, padx=20, pady=20, sticky="nsew")
        
        self.add_student_btn = ctk.CTkButton(self.controls_frame, text="Add New Student", command=self.add_new_student)
        self.add_student_btn.pack(expand=True, padx=20, pady=10, fill="x")

        self.mark_attendance_btn = ctk.CTkButton(self.controls_frame, text="Mark Attendance (Scan Face)", command=self.mark_attendance)
        self.mark_attendance_btn.pack(expand=True, padx=20, pady=10, fill="x")

        self.display_frame = ctk.CTkFrame(self.main_frame)
        self.display_frame.grid(row=0, column=1, rowspan=2, padx=20, pady=20, sticky="nsew")
        
        self.image_label = ctk.CTkLabel(self.display_frame, text="Captured image will be shown here")
        self.image_label.pack(expand=True, padx=10, pady=10)

        self.status_label = ctk.CTkLabel(self, text="Welcome! Initializing...", font=ctk.CTkFont(size=14))
        self.status_label.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        # --- Load Data and Model ---
        self.load_dependencies()

    def load_dependencies(self):
        try:
            # Create embeddings directory if it doesn't exist
            if not os.path.exists(EMBEDDINGS_DIR):
                os.makedirs(EMBEDDINGS_DIR)

            # Load Keras model
            self.embedding_model = tf.keras.models.load_model("face_embedding_model.keras")

            # Load student data CSV first
            if os.path.exists(CSV_FILE):
                self.student_data = pd.read_csv(CSV_FILE)

            # Load all corresponding embeddings
            loaded_embeddings = []
            loaded_labels = []
            for person_id in self.student_data['idlabel']:
                embedding_path = os.path.join(EMBEDDINGS_DIR, f"{person_id}.npz")
                if os.path.exists(embedding_path):
                    data = np.load(embedding_path)
                    loaded_embeddings.append(data['embedding'])
                    loaded_labels.append(person_id)
            
            if loaded_embeddings:
                self.embeddings = np.array(loaded_embeddings)
                self.labels = np.array(loaded_labels, dtype=int)
            
            self.status_label.configure(text=f"✅ Model & {len(self.labels)} student records loaded.")

        except Exception as e:
            messagebox.showerror("Loading Error", f"Failed to load dependencies: {e}")
            self.quit()

    def generate_unique_id(self):
        while True:
            new_id = random.randint(1000, 9999)
            if new_id not in self.student_data['idlabel'].values:
                return new_id

    def preprocess_image(self, img_pil):
        img_resized = img_pil.resize(IMG_SIZE)
        img_array = np.array(img_resized).astype("float32") / 255.0
        return np.expand_dims(img_array, axis=0)

    def add_new_student(self):
        filepath = filedialog.askopenfilename(
            title="Select a clear picture of the student",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not filepath:
            return

        dialog = AddStudentDialog(self)
        student_info = dialog.wait_for_input()
        
        if student_info:
            try:
                person_id = self.generate_unique_id()
                
                img = Image.open(filepath).convert("RGB")
                img_processed = self.preprocess_image(img)
                emb = self.embedding_model.predict(img_processed)[0]

                # **FIX**: Save this person's embedding to its own file
                embedding_path = os.path.join(EMBEDDINGS_DIR, f"{person_id}.npz")
                np.savez(embedding_path, embedding=emb)

                # **FIX**: Update in-memory arrays correctly
                self.embeddings = np.vstack([self.embeddings, emb]) if self.embeddings.size else np.array([emb])
                self.labels = np.append(self.labels, person_id)

                # Update DataFrame and save to CSV
                new_student = pd.DataFrame([{
                    "name": student_info['name'], "rollno": student_info['rollno'],
                    "attendance_so_far": 0, "idlabel": person_id
                }])
                self.student_data = pd.concat([self.student_data, new_student], ignore_index=True)
                self.student_data.to_csv(CSV_FILE, index=False)
                
                self.status_label.configure(text=f"✅ Added {student_info['name']} with ID {person_id}.")
                messagebox.showinfo("Success", f"New student '{student_info['name']}' added successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to add student: {e}")

    def search_face(self, query_emb):
        if self.embeddings.size == 0:
            return None, 0.0
        
        # Compute cosine similarity
        query_emb_norm = np.linalg.norm(query_emb)
        db_norms = np.linalg.norm(self.embeddings, axis=1)
        
        if query_emb_norm == 0 or np.any(db_norms == 0):
            return None, 0.0

        sims = np.dot(self.embeddings, query_emb.T).flatten() / (db_norms * query_emb_norm)
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        
        if best_score >= SIMILARITY_THRESHOLD:
            return self.labels[best_idx], best_score
        else:
            return None, best_score

    def mark_attendance(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            return

        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Camera Error", "Failed to capture image.")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        img_tk = ImageTk.PhotoImage(image=img_pil.resize((320, 240)))
        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk

        try:
            img_processed = self.preprocess_image(img_pil)
            query_emb = self.embedding_model.predict(img_processed)
            
            matched_id, score = self.search_face(query_emb)

            if matched_id is not None:
                # The int() conversion is still good practice, though now types should match
                student_info = self.student_data[self.student_data['idlabel'] == int(matched_id)]
                if not student_info.empty:
                    student_index = student_info.index[0]
                    self.student_data.loc[student_index, 'attendance_so_far'] += 1
                    self.student_data.to_csv(CSV_FILE, index=False)
                    
                    name = student_info.iloc[0]['name']
                    rollno = student_info.iloc[0]['rollno']
                    attendance = self.student_data.loc[student_index, 'attendance_so_far']
                    
                    result_text = f"✅ Match Found! ({score:.2f})\n\nName: {name}\nRoll No: {rollno}\nAttendance Marked! (Total: {attendance})"
                    self.status_label.configure(text=f"Attendance marked for {name}.")
                    messagebox.showinfo("Attendance Success", result_text)
                else:
                    self.status_label.configure(text=f"❌ Error: Matched ID {matched_id} not in CSV.")
            else:
                self.status_label.configure(text=f"❌ No match found. Highest similarity: {score:.2f}")
                messagebox.showwarning("Attendance Failed", f"No match found. Please try again.\n(Highest similarity: {score:.2f})")

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {e}")

if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()