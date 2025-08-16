from faster_whisper import WhisperModel
import torch
import os


def _select_runtime():
    try:
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    device = "cuda" if has_cuda else "cpu"
    model_size = "large-v3" if has_cuda else "small"
    compute_type = "float16" if has_cuda else "int8"
    cpu_threads = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    return model_size, device, compute_type, cpu_threads


def transcribeAudio(audio_path):
    try:
        print("Транскрибирование аудио (уровень сегментов)...")
        model_size, device, compute_type, cpu_threads = _select_runtime()
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)

        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads if device == "cpu" else 0,
            num_workers=2,
        )
        print(f"Модель Faster-Whisper загружена: {model_size} на {device} ({compute_type})")

        segments, info = model.transcribe(
            audio=audio_path,
            beam_size=5,
            language="ru",
            condition_on_previous_text=True,
            vad_filter=True
        )
        segments = list(segments)
        extracted_texts = [[seg.text, float(seg.start), float(seg.end)] for seg in segments]
        return extracted_texts
    except Exception as e:
        print("Ошибка транскрибирования (сегменты):", e)
        return []


def transcribe_segment_word_level(audio_path):
    """Возвращает структуру со словами и таймкодами для анимации субтитров."""
    try:
        print("Загрузка модели Faster-Whisper для словных таймкодов...")
        model_size, device, compute_type, cpu_threads = _select_runtime()
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads if device == "cpu" else 0,
            num_workers=2,
        )
        print(f"Модель загружена: {model_size} на {device} ({compute_type}). Транскрибирование для словных таймкодов...")

        segments_gen, info = model.transcribe(
            audio=audio_path,
            beam_size=5,
            language="ru",
            condition_on_previous_text=True,
            vad_filter=True,
            word_timestamps=True
        )
        result = {"segments": []}
        for seg in segments_gen:
            seg_dict = {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text if isinstance(seg.text, str) else str(seg.text)
            }
            words_arr = []
            if getattr(seg, "words", None):
                for w in seg.words:
                    if w is None:
                        continue
                    start = float(w.start) if getattr(w, "start", None) is not None else float(seg.start)
                    end = float(w.end) if getattr(w, "end", None) is not None else float(seg.end)
                    text = w.word if hasattr(w, "word") else (w.text if hasattr(w, "text") else "")
                    words_arr.append({"start": start, "end": end, "text": text})
            seg_dict["words"] = words_arr
            result["segments"].append(seg_dict)

        if result["segments"]:
            print(f"Сформировано {len(result['segments'])} сегментов со словными таймкодами.")
            return result
        else:
            print("Предупреждение: не получены сегменты со словными таймкодами.")
            return None
    except FileNotFoundError:
        print(f"Ошибка: не найден аудиофайл для транскрибирования: {audio_path}")
        return None
    except Exception as e:
        print(f"Ошибка транскрибирования (слова): {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    audio_path = "audio.wav"
    print("--- Тест транскрибации сегментов ---")
    seg_tr = transcribeAudio(audio_path)
    if seg_tr:
        trans_text = ""
        for text, start, end in seg_tr:
            trans_text += f"{start:.2f} - {end:.2f}: {text}\n"
        print(trans_text)
    else:
        print("Сегментная транскрибация не удалась.")

    print("\n--- Тест словной транскрибации ---")
    word_res = transcribe_segment_word_level(audio_path)
    if word_res:
        print("Ключи результата:", list(word_res.keys()))
        print("Количество сегментов:", len(word_res.get("segments", [])))
        if word_res.get("segments"):
            print("Первые 10 слов первого сегмента:")
            first_words = word_res["segments"][0].get("words", [])
            for i, w in enumerate(first_words[:10]):
                print(f"  {w.get('start', 0):.2f} - {w.get('end', 0):.2f}: {w.get('text','')}")
    else:
        print("Словная транскрибация не удалась.")
