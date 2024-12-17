import tkinter as tk
from tkinter import Toplevel, Label, Button, Canvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
from PIL import Image, ImageTk
import glob
from sklearn.preprocessing import StandardScaler
from FeatureExtraction import FeatureExtraction

class PolkaDotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Polka Dot Detector")

        # Rutas de imágenes
        paths = [
            "/Users/aaronmulvey/Documents/Proyectos ML/Polka Dot Detector/data/datasetLunares/dysplasticNevi/train/",
            "/Users/aaronmulvey/Documents/Proyectos ML/Polka Dot Detector/data/datasetLunares/spitzNevus/train/"
        ]

        # Calcular características y etiquetas
        self.features = []
        self.labels = []

        for label, path in enumerate(paths):
            for filename in glob.glob(path + "*.jpg"):
                img = cv2.imread(filename)
                self.features.append(FeatureExtraction.get_features(img))
                self.labels.append(label)

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        self.labels = 2 * self.labels - 1  # Convertir etiquetas a -1 y 1

        # Gráficos y estado
        self.current_plot = 0
        self.figures = [
            self.create_feature_space,
            self.create_loss_function,
            self.create_hyperplane
        ]
        self.figure_names = ["Feature Space", "Loss Function", "Optimal Hyperplane"]

        # Botones principales
        self.boton_siguiente = Button(root, text="--> Feature Space", command=self.next_plot)
        self.boton_siguiente.pack(pady=5)
        Button(root, text="Predict Polka Dot", command=self.open_prediction_window).pack(pady=5)

        # Espacio para gráficos
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack()
        self.canvas = None
        self.show_plot()

    def show_plot(self):
        # Limpia el canvas anterior
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        # Crea el gráfico actual
        fig = self.figures[self.current_plot]()
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack()
        self.canvas.draw()

        # Actualiza el nombre del botón
        next_index = (self.current_plot + 1) % len(self.figures)
        self.boton_siguiente.config(text=f"--> {self.figure_names[next_index]}")

    def next_plot(self):
        self.current_plot = (self.current_plot + 1) % len(self.figures)
        self.show_plot()

    def create_feature_space(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Feature Espace")

        for i, f in enumerate(self.features):
            ax.scatter(f[0], f[1], f[2], c='k' if self.labels[i] == -1 else 'r')

        ax.set_xlabel('B')
        ax.set_ylabel('G')
        ax.set_zlabel('R')
        return fig

    def create_loss_function(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Loss Function")

        X, Y = np.meshgrid(np.linspace(-6, 6, 30), np.linspace(-6, 6, 30))
        Z = X ** 2 + Y ** 2
        ax.plot_surface(X, Y, Z, cmap="jet", alpha=0.7)

        ax.set_xlabel('w1')
        ax.set_ylabel('w2')
        ax.set_zlabel('Loss')
        return fig

    def create_hyperplane(self):
        # Normalización de características
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(self.features)

        # Cálculo del hiperplano con características normalizadas
        A = np.zeros((4, 4))
        b = np.zeros((4, 1))
        for i, feature_row in enumerate(features_normalized):
            x = np.append([1], feature_row).reshape((4, 1))
            y = self.labels[i]
            A = A + x @ x.T
            b = b + x * y

        invA = np.linalg.inv(A)
        W = np.dot(invA, b)

        # Crear la malla para el hiperplano
        X, Y = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
        Z = -(W[1] * X + W[2] * Y + W[0]) / W[3]

        # Crear la figura
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Normalized Data & Optimal Hyperplane")

        # Dibujar la superficie del hiperplano
        ax.plot_surface(X, Y, Z, cmap="Blues", alpha=0.5)

        # Dibujar los puntos normalizados
        for i, feature_row in enumerate(features_normalized):
            ax.scatter(feature_row[0], feature_row[1], feature_row[2], c='k' if self.labels[i] == -1 else 'r')

        ax.set_xlabel('B (Normalized)')
        ax.set_ylabel('G (Normalized)')
        ax.set_zlabel('R (Normalized)')
        return fig

    # --------------- Predicción -----------------
    def open_prediction_window(self):
        pred_window = Toplevel(self.root)
        pred_window.title("Predict Polka Dot")
        Label(pred_window, text="Select Image to Predict").pack(pady=5)

        # Carpetas con imágenes
        folder1 = "/Users/aaronmulvey/Documents/Proyectos ML/Polka Dot Detector/data/datasetLunares/dysplasticNevi/train"
        folder2 = "/Users/aaronmulvey/Documents/Proyectos ML/Polka Dot Detector/data/datasetLunares/spitzNevus/train"

        # Leer imágenes de las carpetas
        img_paths1 = glob.glob(f"{folder1}/*.jpg")
        img_paths2 = glob.glob(f"{folder2}/*.jpg")

        # Crear el canvas para las miniaturas
        canvas = Canvas(pred_window)
        canvas.pack()

        # Mostrar miniaturas en dos filas
        self.display_image_row(canvas, img_paths1, "Dysplastic Nevi")
        self.display_image_row(canvas, img_paths2, "Spitz Nevus")

    def display_image_row(self, canvas, img_paths, row_label):
        Label(canvas, text=row_label).pack(pady=5)  # Etiqueta para la fila
        frame = tk.Frame(canvas)
        frame.pack(pady=5)

        if not hasattr(self, 'thumbnails'):
            self.thumbnails = []  # Crear lista de miniaturas si no existe

        for img_path in img_paths:
            img = Image.open(img_path).resize((80, 80))  # Tamaño de miniatura
            img_tk = ImageTk.PhotoImage(img)
            self.thumbnails.append(img_tk)  # Mantener referencia a la miniatura

            btn = Button(frame, image=img_tk, command=lambda p=img_path: self.predict_image(p))
            btn.pack(side='left', padx=5)

    def predict_image(self, img_path):
        # Función de predicción para la imagen seleccionada
        img = cv2.imread(img_path)
        features = self.get_features(img)

        # Modelo lineal simple para ejemplo
        prediction = np.sign(0.5 * features[0] + 0.5 * features[1] - 0.5)
        result = "Dysplastic Nevi" if prediction == -1 else "Spitz Nevus"

        # Selección automática del path de la imagen
        if result == "Dysplastic Nevi":
            result_img_path = "/Users/aaronmulvey/Documents/Proyectos ML/Polka Dot Detector/data/PolkaDots/DysplasticNevus.jpg"
        else:
            result_img_path = "/Users/aaronmulvey/Documents/Proyectos ML/Polka Dot Detector/data/PolkaDots/SpitzNevus.jpg"

        # Crear una nueva ventana para mostrar la imagen y la predicción
        pred_window = Toplevel(self.root)
        pred_window.title("Prediction")

        # Cargar la imagen con Pillow
        img_pil = Image.open(result_img_path).resize((200, 200))  # Redimensionar la imagen
        img_tk = ImageTk.PhotoImage(img_pil)

        # Mostrar la imagen
        img_label = Label(pred_window, image=img_tk)
        img_label.image = img_tk  # Guardar referencia para evitar que se elimine
        img_label.pack(pady=10)

        # Mostrar el resultado de la predicción
        result_label = Label(pred_window, text=f"{result}", font=("Helvetica", 16))
        result_label.pack(pady=10)

    def get_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        mask = np.uint8(1 * (gray < threshold))
        B = np.sum(img[:, :, 0] * mask) / (255 * np.sum(mask))
        G = np.sum(img[:, :, 1] * mask) / (255 * np.sum(mask))
        R = np.sum(img[:, :, 2] * mask) / (255 * np.sum(mask))
        return [B, G, R]



# Main
if __name__ == "__main__":
    root = tk.Tk()
    app = PolkaDotApp(root)
    root.mainloop()
