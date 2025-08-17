import os
import subprocess
import glob
import time


def download_youtube_video(url):
    """
    Скачивает видео с YouTube устойчивым способом через yt-dlp (Python API), с автоматическим фолбэком
    на pytubefix/pytube. Гарантируется итоговый MP4 без перекодирования (ffmpeg -c copy).
    Возвращает путь к мастер-файлу MP4 или None при ошибке.
    """
    try:
        out_dir = "videos"
        os.makedirs(out_dir, exist_ok=True)

        def _ffmpeg_merge(v, a, merged_final):
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
            try:
                if os.path.exists(merged_final):
                    os.remove(merged_final)
                os.replace(tmp_out, merged_final)
                return merged_final
            except Exception as ren_e:
                print(f"Ошибка при переименовании итогового файла: {ren_e}")
                return None

        def _attempt_local_merge(start_time):
            video_parts = [
                p for p in glob.glob(os.path.join(out_dir, "master-*.mp4"))
                if os.path.getmtime(p) >= (start_time - 1.0)
            ]
            audio_parts = [
                p for p in glob.glob(os.path.join(out_dir, "master-*.m4a"))
                if os.path.getmtime(p) >= (start_time - 1.0)
            ]

            def stem(path):
                base = os.path.basename(path)
                return os.path.splitext(base)[0]  # master-<id>

            video_map = {stem(p): p for p in video_parts}
            audio_map = {stem(p): p for p in audio_parts}
            common = sorted(
                set(video_map.keys()) & set(audio_map.keys()),
                key=lambda s: os.path.getmtime(video_map[s]),
                reverse=True
            )

            if common:
                key = common[0]
                v = video_map[key]
                a = audio_map[key]
                merged_final = os.path.join(out_dir, f"{key}.mp4")
                res = _ffmpeg_merge(v, a, merged_final)
                if res:
                    # Чистим части
                    try:
                        os.remove(v)
                    except Exception:
                        pass
                    try:
                        os.remove(a)
                    except Exception:
                        pass
                    print(f"Загрузка через yt-dlp завершена. Итоговый файл: {res}")
                    return res
            return None

        def _fallback_pytube(src_url):
            try:
                from pytubefix import YouTube
                lib = "pytubefix"
            except Exception:
                try:
                    from pytube import YouTube
                    lib = "pytube"
                except Exception as e_imp:
                    print(f"Фолбэк pytubefix/pytube недоступен: {e_imp}")
                    return None

            try:
                print(f"Фолбэк: {lib} — пытаюсь скачать прогрессивный MP4…")
                # Имитируем поведение WEB‑клиента (для pytubefix) и отключаем OAuth/кэш.
                if lib == "pytubefix":
                    yt = YouTube(src_url, client="WEB", use_oauth=False, allow_oauth_cache=False)
                else:
                    yt = YouTube(src_url, use_oauth=False, allow_oauth_cache=False)
                # Прогрессивный (видео+аудио в одном mp4)
                stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
                if stream:
                    filename = f"master-{yt.video_id}.mp4"
                    final_path = stream.download(output_path=out_dir, filename=filename)
                    print(f"Загрузка завершена через {lib}. Итоговый файл: {final_path}")
                    return final_path

                print("Прогрессивный MP4 недоступен. Пытаюсь скачать адаптивные потоки mp4+m4a…")
                vstream = yt.streams.filter(adaptive=True, only_video=True, file_extension="mp4").order_by("resolution").desc().first()
                astream = yt.streams.filter(adaptive=True, only_audio=True, mime_type="audio/mp4").order_by("abr").desc().first()
                if not vstream or not astream:
                    print("Не удалось найти подходящие адаптивные потоки mp4 (видео) и m4a (аудио) для слияния.")
                    return None

                vpath = vstream.download(output_path=out_dir, filename=f"master-{yt.video_id}-video")
                apath = astream.download(output_path=out_dir, filename=f"master-{yt.video_id}-audio")
                merged_final = os.path.join(out_dir, f"master-{yt.video_id}.mp4")
                res = _ffmpeg_merge(vpath, apath, merged_final)
                if res:
                    try:
                        os.remove(vpath)
                    except Exception:
                        pass
                    try:
                        os.remove(apath)
                    except Exception:
                        pass
                    print(f"Загрузка завершена через {lib}. Итоговый файл: {res}")
                    return res
                return None
            except Exception as e_fb:
                print(f"Ошибка фолбэка {lib}: {e_fb}")
                return None

        start_time = time.time()

        # Попытка 1: yt-dlp (Python API)
        try:
            import yt_dlp as ytdlp
        except Exception as imp_err:
            print(f"yt-dlp (Python API) недоступен: {imp_err}")
            print("Перехожу к фолбэку pytubefix/pytube…")
            return _fallback_pytube(url)

        try:
            print("Использую yt-dlp (Python API) для устойчивой загрузки (mp4+m4a, без перекодирования)…")
            format_str = "bv*[ext=mp4][vcodec^=avc1]+ba[ext=m4a]/bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b"
            out_tmpl = os.path.join(out_dir, "master-%(id)s.%(ext)s")
            ydl_opts = {
                "format": format_str,
                "outtmpl": out_tmpl,
                "merge_output_format": "mp4",
                "noplaylist": True,
                "prefer_ffmpeg": True,
                "quiet": True,
                "no_warnings": True,
                "retries": 3,
                # yt-dlp по умолчанию использует -c copy для merge; добавим faststart
                "postprocessor_args": ["-movflags", "+faststart"],
            }
            with ytdlp.YoutubeDL(ydl_opts) as ydl:
                _ = ydl.extract_info(url, download=True)

            # Проверяем итоговый mp4
            mp4s = [
                p for p in glob.glob(os.path.join(out_dir, "master-*.mp4"))
                if os.path.getmtime(p) >= (start_time - 1.0)
            ]
            if mp4s:
                mp4s.sort(key=os.path.getmtime, reverse=True)
                final_path = mp4s[0]
                print(f"Загрузка через yt-dlp завершена. Итоговый файл: {final_path}")
                return final_path

            # Пробуем локальное слияние, если остались отдельные дорожки
            merged = _attempt_local_merge(start_time)
            if merged:
                return merged

            print("yt-dlp не создал финальный MP4 и не найдены пары mp4+m4a для слияния.")
            print("Перехожу к фолбэку pytubefix/pytube…")
            return _fallback_pytube(url)

        except Exception as e_ydl:
            print(f"Ошибка при загрузке через yt-dlp (Python API): {e_ydl}")
            print("Перехожу к фолбэку pytubefix/pytube…")
            return _fallback_pytube(url)

    except Exception as e:
        print(f"Непредвиденная ошибка при скачивании: {e}")
        print("Не удалось скачать видео. Проверьте доступность ffmpeg в PATH.")
        return None


if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ")
    downloaded_file = download_youtube_video(youtube_url)
    if downloaded_file:
        print(f"\nDownload finished. File available at: {downloaded_file}")
    else:
        print("\nDownload failed.")
