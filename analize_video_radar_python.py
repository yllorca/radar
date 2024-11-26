import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


class BioradarAnalyzer:
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=50,
            detectShadows=False
        )

    def process_video(self, video_path):
        """
        Procesa el video del bioradar y extrae información sobre movimiento
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("No se pudo abrir el video")

        trajectories = []
        motion_data = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocesamiento
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Detección de movimiento
            mask = self.background_subtractor.apply(blur)

            # Eliminación de ruido
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Encontrar contornos de movimiento
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            current_frame_trajectories = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    center = (x + w // 2, y + h // 2)
                    current_frame_trajectories.append(center)

                    if len(trajectories) > 0:
                        prev_centers = trajectories[-1]
                        for prev_center in prev_centers:
                            dist = np.sqrt((center[0] - prev_center[0]) ** 2 +
                                           (center[1] - prev_center[1]) ** 2)
                            if dist < 50:  # Umbral de seguimiento
                                motion_data.append({
                                    'frame': frame_count,
                                    'position_x': center[0],
                                    'position_y': center[1],
                                    'velocity': dist,
                                    'direction': np.arctan2(center[1] - prev_center[1],
                                                            center[0] - prev_center[0])
                                })

            trajectories.append(current_frame_trajectories)
            frame_count += 1

        cap.release()
        return self.analyze_motion_data(motion_data), motion_data

    def analyze_motion_data(self, motion_data):
        if not motion_data:
            return {"error": "No se detectó movimiento significativo"}

        velocities = [d['velocity'] for d in motion_data]
        directions = [d['direction'] for d in motion_data]

        stats = {
            'velocidad_promedio': np.mean(velocities),
            'velocidad_max': np.max(velocities),
            'direccion_predominante': self._get_predominant_direction(directions),
            'num_detecciones': len(motion_data)
        }

        if len(velocities) > 1:
            fs = 30  # Frecuencia de muestreo asumida
            f, Pxx = signal.welch(velocities, fs, nperseg=min(256, len(velocities)))
            peak_freq = f[np.argmax(Pxx)]
            stats['frecuencia_movimiento'] = peak_freq

        return stats

    def _get_predominant_direction(self, directions):
        mean_direction = np.mean(directions)
        cardinal_directions = {
            'Norte': (-np.pi / 4, np.pi / 4),
            'Este': (-3 * np.pi / 4, -np.pi / 4),
            'Sur': (3 * np.pi / 4, np.pi),
            'Oeste': (np.pi / 4, 3 * np.pi / 4)
        }

        for direction, (min_angle, max_angle) in cardinal_directions.items():
            if min_angle <= mean_direction <= max_angle:
                return direction
        return 'Indeterminada'

    def visualize_trajectories(self, results, output_path):
        """
        Genera una visualización de las trayectorias detectadas
        """
        stats, motion_data = results  # Ahora recibimos tanto las estadísticas como los datos de movimiento

        plt.figure(figsize=(12, 8))

        # Graficar trayectorias
        x_positions = [d['position_x'] for d in motion_data]
        y_positions = [d['position_y'] for d in motion_data]
        velocities = [d['velocity'] for d in motion_data]

        # Crear scatter plot con colores basados en velocidad
        scatter = plt.scatter(x_positions, y_positions,
                              c=velocities,
                              cmap='viridis',
                              alpha=0.5,
                              s=10)

        plt.colorbar(scatter, label='Velocidad')

        # Añadir flechas de dirección cada N puntos
        step = len(motion_data) // 20  # Mostrar ~20 flechas
        if step < 1:
            step = 1

        for i in range(0, len(motion_data), step):
            d = motion_data[i]
            vel = d['velocity']
            dir = d['direction']
            dx = vel * np.cos(dir) * 0.5  # Factor 0.5 para hacer las flechas más cortas
            dy = vel * np.sin(dir) * 0.5
            plt.arrow(d['position_x'], d['position_y'],
                      dx, dy,
                      head_width=5,
                      head_length=10,
                      fc='r',
                      ec='r',
                      alpha=0.3)

        plt.title('Trayectorias Detectadas')
        plt.xlabel('Posición X')
        plt.ylabel('Posición Y')
        plt.grid(True)

        # Añadir estadísticas como texto
        stats_text = f"Velocidad promedio: {stats['velocidad_promedio']:.2f}\n"
        stats_text += f"Velocidad máxima: {stats['velocidad_max']:.2f}\n"
        stats_text += f"Dirección predominante: {stats['direccion_predominante']}\n"
        stats_text += f"Número de detecciones: {stats['num_detecciones']}"

        plt.figtext(0.02, 0.02, stats_text, fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8))

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
