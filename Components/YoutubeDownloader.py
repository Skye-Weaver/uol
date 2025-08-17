import os
import subprocess
import glob
import time


def download_youtube_video(url):
    """
    Скачивает видео с YouTube устойчивым способом через yt-dlp, гарантируя итоговый MP4 без перекодирования.
    Возвращает путь к мастер-файлу MP4 или None при ошибке.
    """
    try:
        out_dir = "videos"
        os.makedirs(out_dir, exist_ok=True)

        # Основной путь: yt-dlp CLI, без интерактива
        print("Использую yt-dlp для устойчивой загрузки (mp4+m4a, без перекодирования)…")

        format_str = "bv*[ext=mp4][vcodec^=avc1]+ba[ext=m4a]/bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b"
        out_tmpl = os.path.join(out_dir, "master-%(id)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "-f", format_str,
            "--merge-output-format", "mp4",
            "-o", out_tmpl,
            "--no-playlist",
            "--no-warnings",
            "--prefer-ffmpeg",
            url,
        ]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("Ошибка: yt-dlp завершился с ненулевым кодом возврата.")
            print(f"Код возврата: {result.returncode}")
            if result.stderr:
                print("stderr yt-dlp:")
                print(result.stderr.strip())
            print("Команда:", " ".join(cmd))
            print("Не удалось скачать видео даже через yt-dlp. Проверьте доступность URL и ffmpeg в PATH.")
            return None

        # Поиск созданного мастер-файла mp4 (noplaylist=True -> один файл)
        candidates = [
            p for p in glob.glob(os.path.join(out_dir, "master-*.mp4"))
            if os.path.getmtime(p) >= (start_time - 1.0)
        ]

        if len(candidates) >= 1:
            candidates.sort(key=os.path.getmtime, reverse=True)
            final_path = candidates[0]
            print(f"Загрузка через yt-dlp завершена. Итоговый файл: {final_path}")
            return final_path

        # Редкий случай: yt-dlp не объединил дорожки автоматически — пробуем быстрый merge
        video_parts = [
            p for p in glob.glob(os.path.join(out_dir, "master-*.mp4"))
            if os.path.getmtime(p) >= (start_time - 1.0)
        ]
        audio_parts = [
            p for p in glob.glob(os.path.join(out_dir, "master-*.m4a"))
            if os.path.getmtime(p) >= (start_time - 1.0)
        ]

        # Сопоставляем по префиксу 'master-<id>'
        def stem(path):
            base = os.path.basename(path)
            return os.path.splitext(base)[0]  # master-<id>

        video_map = {stem(p): p for p in video_parts}
        audio_map = {stem(p): p for p in audio_parts}
        common = sorted(set(video_map.keys()) & set(audio_map.keys()), key=lambda s: os.path.getmtime(video_map[s]), reverse=True)

        if common:
            key = common[0]
            v = video_map[key]
            a = audio_map[key]
            merged_final = os.path.join(out_dir, f"{key}.mp4")
            tmp_out = merged_final + ".tmp"

            print("Автослияние дорожек через ffmpeg (без перекодирования, copy)…")
            merge_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "info",
                "-y",
                "-i", v,
                "-i", a,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "copy",
                "-movflags", "+faststart",
                "-shortest",
                tmp_out,
            ]
            print("Запуск команды слияния:", " ".join(merge_cmd))
            mres = subprocess.run(merge_cmd, capture_output=True, text=True)
            if mres.returncode != 0:
                print("Ошибка: ffmpeg завершился с ненулевым кодом возврата.")
                print(f"Код возврата: {mres.returncode}")
                if mres.stderr:
                    print("stderr ffmpeg:")
                    print(mres.stderr.strip())
                print("Команда:", " ".join(merge_cmd))
                print("Не удалось объединить дорожки. Проверьте ffmpeg в PATH.")
                return None

            # Переименовываем временный файл в финальный
            try:
                if os.path.exists(merged_final):
                    os.remove(merged_final)
                os.replace(tmp_out, merged_final)
                # Чистим части
                try:
                    os.remove(v)
                except Exception:
                    pass
                try:
                    os.remove(a)
                except Exception:
                    pass
                print(f"Загрузка через yt-dlp завершена. Итоговый файл: {merged_final}")
                return merged_final
            except Exception as ren_e:
                print(f"Ошибка при переименовании итогового файла: {ren_e}")
                return None

        print("yt-dlp не создал финальный MP4 и не найдены пары mp4+m4a для слияния.")
        return None

    except Exception as e:
        print(f"Непредвиденная ошибка при скачивании: {e}")
        print("Не удалось скачать видео даже через yt-dlp. Проверьте доступность URL и ffmpeg в PATH.")
        return None


if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ")
    downloaded_file = download_youtube_video(youtube_url)
    if downloaded_file:
        print(f"\nDownload finished. File available at: {downloaded_file}")
    else:
        print("\nDownload failed.")
