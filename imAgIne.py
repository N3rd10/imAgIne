import tkinter as tk
from tkinter import ttk
import time
# Splash screen class
class SplashScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.geometry("350x120+500+300")
        self.root.configure(bg="#1E1E1E")
        ttk.Style().theme_use("clam")
        ttk.Label(self.root, text="imAgIne is loading...", font=("Segoe UI", 12, "bold"), background="#1E1E1E", foreground="#D4D4D4").pack(pady=10)
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)
        self.percent_label = ttk.Label(self.root, text="0%", background="#1E1E1E", foreground="#D4D4D4")
        self.percent_label.pack()
        self.root.update()

    def update(self, percent):
        self.progress["value"] = percent
        self.percent_label.config(text=f"{percent}%")
        self.root.update_idletasks()

    def close(self):
        self.root.destroy()

splash = SplashScreen()

splash.update(5)
time.sleep(0.15)
import tkinter.scrolledtext as scrolledtext
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox


splash.update(10)
time.sleep(0.15)
import threading

splash.update(15)
time.sleep(0.15)
import tensorflow as tf

splash.update(20)
time.sleep(0.15)
from keras.models import Sequential

splash.update(25)
time.sleep(0.15)
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

splash.update(30)
time.sleep(0.15)
from keras.preprocessing import image_dataset_from_directory

splash.update(35)
time.sleep(0.15)
from keras.callbacks import EarlyStopping
from imagine import settings as imagine_settings
from imagine import utils as imagine_utils
from imagine import train as imagine_train
from imagine import predict as imagine_predict
from imagine import convert as imagine_convert

splash.update(40)
time.sleep(0.15)
import numpy as np

splash.update(45)
time.sleep(0.15)
from PIL import Image

splash.update(50)
time.sleep(0.15)
import os

splash.update(55)
time.sleep(0.15)
import shutil

splash.update(60)
time.sleep(0.15)
import random

splash.update(65)
time.sleep(0.15)
import json

splash.update(70)
time.sleep(0.15)
import sys

import functools

splash.update(80)
time.sleep(0.15)
# Model loading
try:
    default_model_path = "model.h5"
    default_classes_path = "model_classes.json"
    if os.path.exists(default_model_path):
        splash.update(85)
        model_predict = tf.keras.models.load_model(default_model_path)
        splash.update(90)
        if os.path.exists(default_classes_path):
            with open(default_classes_path, "r") as f:
                class_names = json.load(f)
        else:
            class_names = []
    else:
        model_predict = None
        class_names = []
except:
    model_predict = None
    class_names = []

splash.update(100)
time.sleep(0.2)


class DummyStream:
    def write(self, _): pass
    def flush(self): pass

if getattr(sys, 'frozen', False):
    sys.stdout = DummyStream()
    sys.stderr = DummyStream()

# ---------------------- SETTINGS ----------------------
SETTINGS_FILE = "settings.json"
settings = imagine_settings.settings

# ---------------------- MODEL LOAD FOR PREDICTION ----------------------
# Default model placeholder handled by imagine.predict module
model_predict = None
class_names = []

# ---------------------- MAIN WINDOW ----------------------
splash.close()
root = tk.Tk()
root.title("imAgIne") 
root.geometry("850x650")
root.minsize(800, 600)
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller .exe"""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

root.iconbitmap(resource_path("ImAgIne.ico"))



style = ttk.Style()



# Create custom theme and ensure fallback to a valid theme

if settings["theme"] not in style.theme_names():
    settings["theme"] = "vscode-dark"
style.theme_use(settings["theme"])

# ---------------------- SETTINGS MENU ----------------------
def open_settings():
    settings_win = tk.Toplevel(root)
    settings_win.title("Settings")
    settings_win.grab_set()
    settings_win.resizable(False, False)
    settings_win.configure(padx=15, pady=15)

    ttk.Label(settings_win, text="Theme:").grid(row=0, column=0, sticky="w", pady=5)
    theme_var = tk.StringVar(value=settings["theme"])
    theme_combo = ttk.Combobox(settings_win, textvariable=theme_var, values=style.theme_names(), state="readonly")
    theme_combo.grid(row=0, column=1, pady=5)

    ttk.Label(settings_win, text="Default Epochs:").grid(row=1, column=0, sticky="w", pady=5)
    epochs_var = tk.IntVar(value=settings["epochs"])
    ttk.Spinbox(settings_win, from_=1, to=100, textvariable=epochs_var, width=5).grid(row=1, column=1, pady=5)

    def save_and_close():
        settings["theme"] = theme_var.get()
        settings["epochs"] = epochs_var.get()
        # Safe apply: if somehow missing, fall back to vscode-dark
        if settings["theme"] not in style.theme_names():
            settings["theme"] = "vscode-dark"
            style.theme_use(settings["theme"])
        imagine_settings.save_settings()
        settings_win.destroy()  

    ttk.Button(settings_win, text="Save", command=save_and_close).grid(row=2, column=0, columnspan=2, pady=10)

menubar = tk.Menu(root)
root.config(menu=menubar)
settings_menu = tk.Menu(menubar, tearoff=0)
settings_menu.add_command(label="Preferences...", command=open_settings)
menubar.add_cascade(label="Settings", menu=settings_menu)

# ---------------------- NOTEBOOK ----------------------
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both', padx=5, pady=5)

train_frame = ttk.Frame(notebook, padding = 10)
predict_frame = ttk.Frame(notebook, padding = 10)
convert_frame = ttk.Frame(notebook, padding = 10)

notebook.add(train_frame, text="Train Model")
notebook.add(predict_frame, text="Predict Image")
notebook.add(convert_frame, text="Convert Model")

# ---------------------- TRAINING FUNCTIONS ----------------------
folder_path = tk.StringVar()

def auto_split_dataset(base_path, train_ratio=0.8):
    return imagine_utils.auto_split_dataset(base_path, train_ratio)

def choose_folder():
    selected = filedialog.askdirectory()
    if selected:
        folder_path.set(selected)
        log_box.insert(tk.END, f"Selected folder: {selected}\n")
        status_bar.config(text=f"Dataset folder selected: {selected}")

def prompt_save_model():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".h5",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
        title="Save Trained Model As"
    )
    return file_path

# Smooth per-batch progress callback
class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_widget, label_widget, total_epochs, steps_per_epoch):
        super().__init__()
        self.progress_widget = progress_widget
        self.label_widget = label_widget
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.current_step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.current_step += 1
        percent = int((self.current_step / self.total_steps) * 100)
        self.progress_widget["value"] = percent
        self.label_widget.config(text=f"Progress: {percent}%")
        self.progress_widget.update_idletasks()
        self.label_widget.update_idletasks()

class PerformanceLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_widget):
        super().__init__()
        self.log_widget = log_widget

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = (
            f"Epoch {epoch + 1}: "
            f"loss={logs.get('loss', 0):.4f}, "
            f"accuracy={logs.get('accuracy', 0):.4f}, "
            f"val_loss={logs.get('val_loss', 0):.4f}, "
            f"val_accuracy={logs.get('val_accuracy', 0):.4f}"
        )
        self.log_widget.insert(tk.END, msg + "\n")
        self.log_widget.see(tk.END)  # auto-scroll

    def on_train_end(self, logs=None):
        self.log_widget.insert(tk.END, "Training complete.\n")
        self.log_widget.see(tk.END)

def train_model(log_widget):
    try:
        base_path = folder_path.get()
        if not base_path:
            messagebox.showwarning("Missing Folder", "Please select a training folder first.")
            return

        progress["value"] = 0
        progress_label.config(text="Progress: 0%")

        # Delegate training to imagine.train
        def log_cb(text):
            log_widget.insert(tk.END, text + "\n")
            log_widget.see(tk.END)

        def save_path_cb():
            return prompt_save_model()

        log_widget.insert(tk.END, "Starting training via imagine.train...\n")
        imagine_train.train_model(
            base_path,
            log_callback=log_cb,
            progress_callback=None,
            save_path_callback=save_path_cb
        )

    except Exception as e:
        log_widget.insert(tk.END, f"Error: {e}\n")
        messagebox.showerror("Error", str(e))
        status_bar.config(text="Error during training")

def start_training(log_widget):
    threading.Thread(target=train_model, args=(log_widget,), daemon=True).start()

# ---------------------- PREDICTION FUNCTIONS ----------------------
current_model_path = tk.StringVar(value="No model loaded")

def load_model_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
        title="Select Model File"
    )
    if file_path:
        try:
            global model_predict, class_names
            imagine_predict.load_model(file_path)
            model_predict = imagine_predict.model_predict
            class_names = imagine_predict.class_names
            if class_names:
                result_label.config(text=f"Model loaded.\nClasses: {class_names}")
            else:
                messagebox.showwarning("Class Names Missing", "No class list found for this model.")
                result_label.config(text="Model loaded, but no class names found.")

            current_model_path.set(f"Loaded: {os.path.basename(file_path)}")
            status_bar.config(text=f"Model loaded: {file_path}")
        except Exception as e:
            result_label.config(text=f"Error loading model: {e}")
            status_bar.config(text="Error loading model")

def load_and_preprocess_image(path):
    return imagine_utils.load_and_preprocess_image(path, target_size=(224,224))

def choose_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        result = classify_image(file_path)
        result_label.config(text=result)
        status_bar.config(text=f"Image classified: {file_path}")

def classify_image(path):
    try:
        if not imagine_predict.model_predict:
            return "No model loaded. Please load a model first."
        predicted_class, confidence = imagine_predict.classify_image(path)
        return f"Prediction: {predicted_class} ({confidence:.2%} confidence)"
    except Exception as e:
        return f"Error: {e}"



# ---------------------- CONVERSION FUNCTIONS ----------------------

def convert_h5_to_limelight_tflite(h5_path, output_dir, representative_data_gen, quantize=True, log_callback=None):
    return imagine_convert.convert_h5_to_tflite(h5_path, output_dir, representative_data_gen, quantize, log_callback)

def representative_data_gen(val_path):
    image_paths = []
    for class_folder in os.listdir(val_path):
        class_path = os.path.join(val_path, class_folder)
        if os.path.isdir(class_path):
            image_paths += [os.path.join(class_path, fname) for fname in os.listdir(class_path) if fname.lower().endswith((".jpg", ".png"))]

    for path in image_paths[:100]:
        try:
            img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
            yield [img_array]
        except Exception as e:
            print(f"Skipping {path}: {e}")

def run_model_conversion():
    h5_path = h5_model_path.get()
    output_dir = output_folder_path.get()
    rep_path = rep_data_folder_path.get()
    def log_to_ui(msg):
        conversion_log_box.insert(tk.END, msg + "\n")
        conversion_log_box.see(tk.END)

    try:
        convert_h5_to_limelight_tflite(
            h5_path=h5_path,
             output_dir=output_dir,
            # pass a zero-arg callable that yields representative data
            representative_data_gen=functools.partial(representative_data_gen, rep_path),
            quantize=True,
            log_callback=log_to_ui
        )
    except Exception as e:
        log_to_ui(f"Error: {e}")


# ---------------------- TRAINING TAB UI ----------------------
folder_group = ttk.LabelFrame(train_frame, text="Dataset Selection", padding=10)
folder_group.pack(fill='x', pady=5)

ttk.Label(folder_group, text="Training Data Folder:").grid(row=0, column=0, sticky="w")
folder_path = tk.StringVar()
ttk.Entry(folder_group, textvariable=folder_path, width=50).grid(row=0, column=1, padx=5)
ttk.Button(folder_group, text="Browse", command=choose_folder).grid(row=0, column=2)

train_controls = ttk.LabelFrame(train_frame, text="Training Controls", padding=10)
train_controls.pack(fill='x', pady=5)

ttk.Button(train_controls, text="Start Training", command=lambda: start_training(log_box)).pack(pady=5)

progress = ttk.Progressbar(train_controls, orient="horizontal", length=400, mode="determinate")
progress.pack(pady=5)
progress_label = ttk.Label(train_controls, text="Progress: 0%")
progress_label.pack()

log_group = ttk.LabelFrame(train_frame, text="Training Log", padding=5)
log_group.pack(fill='both', expand=True, pady=5)
log_box = scrolledtext.ScrolledText(log_group, wrap=tk.WORD, font=("Consolas", 10))
log_box.pack(fill='both', expand=True)

# ---------------------- PREDICTION TAB UI ----------------------
predict_controls = ttk.LabelFrame(predict_frame, text="Prediction Controls", padding=10)
predict_controls.pack(fill='x', pady=5)

current_model_path = tk.StringVar(value="No model loaded")
ttk.Label(predict_controls, textvariable=current_model_path, foreground="blue").pack(anchor="w", pady=2)
ttk.Button(predict_controls, text="Load Model", command=load_model_file).pack(pady=2)
ttk.Button(predict_controls, text="Browse Image", command=choose_image).pack(pady=2)

result_group = ttk.LabelFrame(predict_frame, text="Prediction Result", padding=10)
result_group.pack(fill='both', expand=True, pady=5)
result_label = ttk.Label(result_group, text="", font=("Arial", 12), wraplength=500)
result_label.pack(anchor="center", pady=10)

# ---------------------- MODEL CONVERSION TAB UI ----------------------
conversion_group = ttk.LabelFrame(convert_frame, text="Model Conversion", padding=10)
conversion_group.pack(fill='x', pady=5)

ttk.Label(conversion_group, text="Keras .h5 Model File:").grid(row=0, column=0, sticky="w")
h5_model_path = tk.StringVar()
ttk.Entry(conversion_group, textvariable=h5_model_path, width=50).grid(row=0, column=1, padx=5)
ttk.Button(conversion_group, text="Browse", command=lambda: h5_model_path.set(filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")]))).grid(row=0, column=2)

ttk.Label(conversion_group, text="Output Folder:").grid(row=1, column=0, sticky="w")
output_folder_path = tk.StringVar()
ttk.Entry(conversion_group, textvariable=output_folder_path, width=50).grid(row=1, column=1, padx=5)
ttk.Button(conversion_group, text="Browse", command=lambda: output_folder_path.set(filedialog.askdirectory())).grid(row=1, column=2)

ttk.Label(conversion_group, text="Representative Data Folder:").grid(row=2, column=0, sticky="w")
rep_data_folder_path = tk.StringVar()
ttk.Entry(conversion_group, textvariable=rep_data_folder_path, width=50).grid(row=2, column=1, padx=5)
ttk.Button(
    conversion_group,
    text="Browse",
    command=lambda: rep_data_folder_path.set(filedialog.askdirectory())
).grid(row=2, column=2)


ttk.Button(conversion_group, text="Convert to Limelight Format", 
           command=run_model_conversion).grid(row=3, column=0, columnspan=4, pady=10)

conversion_log_group = ttk.LabelFrame(convert_frame, text="Conversion Log", padding=5)
conversion_log_group.pack(fill='both', expand=True, pady=5)
conversion_log_box = scrolledtext.ScrolledText(conversion_log_group, wrap=tk.WORD, font=("Consolas", 10))
conversion_log_box.pack(fill='both', expand=True)

# ---------------------- STATUS BAR ----------------------
status_bar = ttk.Label(root, text="Ready", relief="sunken", anchor="w")
status_bar.pack(side="bottom", fill="x")
 
root.mainloop()