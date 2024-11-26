import os

from analize_video_radar_python import BioradarAnalyzer
import argparse



def main():
    # Configurar el parser de argumentos con valores por defecto
    parser = argparse.ArgumentParser(description='Analizar video de Bioradar')
    parser.add_argument('--video', type=str,
                        default='video/video_master.mp4',
                        help='Ruta al archivo de video a analizar')
    parser.add_argument('--output', type=str,
                        default='trayectorias.png',
                        help='Ruta donde guardar la visualización')

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: No se encontró el video en {args.video}")
        print("Asegúrate de que el video existe en la carpeta correcta.")
        return

    try:
        print("Iniciando análisis...")
        analyzer = BioradarAnalyzer()

        print(f"Procesando video: {args.video}")
        results = analyzer.process_video(args.video)

        print("\nResultados del análisis:")
        stats = results[0]  # Las estadísticas están en la primera posición
        for key, value in stats.items():
            print(f"{key}: {value}")

        print(f"\nGenerando visualización en: {args.output}")
        analyzer.visualize_trajectories(results, args.output)

        print("\nAnálisis completado exitosamente!")

    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")
        raise  # Esto nos mostrará el traceback completo para debugging


if __name__ == "__main__":
    main()
