import os
import subprocess


def download_youtube_video(url):
    """
    Скачивает видео с YouTube используя только pytubefix с имитацией веб-клиента.
    Автоматически выбирает лучший доступный поток и возвращает путь к MP4 файлу.
    Возвращает полный путь к скачанному MP4-файлу или None в случае ошибки.
    """
    try:
        from pytubefix import YouTube

        out_dir = "videos"
        os.makedirs(out_dir, exist_ok=True)

        def _ffmpeg_merge(video_path, audio_path, output_path):
            """Сливает видео и аудио потоки в один MP4 файл без перекодирования."""
            print("Слияние потоков через ffmpeg...")
            merge_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", video_path,
                "-i", audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "copy",
                "-movflags", "+faststart",
                output_path
            ]

            result = subprocess.run(merge_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Ошибка ffmpeg: {result.stderr}")
                return False

            # Удаляем временные файлы
            try:
                os.remove(video_path)
                os.remove(audio_path)
            except Exception as e:
                print(f"Предупреждение: не удалось удалить временные файлы: {e}")

            return True

        print("Инициализация YouTube с веб-клиентом...")
        # Инициализируем с имитацией веб-клиента для обхода защиты
        yt = YouTube(url, client="WEB")

        print(f"Заголовок видео: {yt.title}")
        print("Поиск доступных потоков...")

        # Сначала пытаемся найти прогрессивный MP4 поток (видео + аудио)
        progressive_stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()

        if progressive_stream:
            print(f"Найден прогрессивный поток: {progressive_stream.resolution}")
            filename = f"master-{yt.video_id}.mp4"
            final_path = progressive_stream.download(output_path=out_dir, filename=filename)
            print(f"Загрузка завершена. Файл: {final_path}")
            return final_path

        print("Прогрессивные потоки недоступны. Использую адаптивные потоки...")

        # Ищем лучший видео поток (только видео)
        video_stream = yt.streams.filter(adaptive=True, only_video=True, file_extension="mp4").order_by("resolution").desc().first()

        # Ищем лучший аудио поток (M4A)
        audio_stream = yt.streams.filter(adaptive=True, only_audio=True, mime_type="audio/mp4").order_by("abr").desc().first()

        if not video_stream or not audio_stream:
            print("Не найдены подходящие адаптивные потоки для слияния")
            return None

        print(f"Видео поток: {video_stream.resolution}")
        print(f"Аудио поток: {audio_stream.abr}")

        # Скачиваем видео и аудио отдельно
        video_path = video_stream.download(output_path=out_dir, filename=f"master-{yt.video_id}-video")
        audio_path = audio_stream.download(output_path=out_dir, filename=f"master-{yt.video_id}-audio")

        # Сливаем в финальный файл
        final_path = os.path.join(out_dir, f"master-{yt.video_id}.mp4")

        if _ffmpeg_merge(video_path, audio_path, final_path):
            print(f"Загрузка и слияние завершены. Файл: {final_path}")
            return final_path
        else:
            print("Ошибка при слиянии потоков")
            return None

    except Exception as e:
        print(f"Ошибка при скачивании видео: {e}")
        return None


if __name__ == "__main__":
    # Пример использования (замените URL на реальный)
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Пример URL
    downloaded_file = download_youtube_video(youtube_url)
    if downloaded_file:
        print(f"\nЗагрузка завершена. Файл доступен: {downloaded_file}")
    else:
        print("\nЗагрузка не удалась.")
