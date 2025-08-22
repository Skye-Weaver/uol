# Исправление проблем с загрузкой YouTube видео

## Проблема
YouTube блокирует запросы без аутентификации, требуя cookies или другие методы аутентификации.

## Решения

### Решение 1: Использование cookies (Рекомендуется)

1. **Установите расширение для браузера:**
   - Для Chrome: [Get cookies.txt](https://chrome.google.com/webstore/detail/get-cookiestxt/bgaddhkoddajcdgaggjhjhaoccdenhg)
   - Для Firefox: [Export Cookies](https://addons.mozilla.org/en-US/firefox/addon/export-cookies-txt/)

2. **Получите cookies:**
   - Зайдите на YouTube в браузере
   - Войдите в свой аккаунт Google
   - Используйте расширение для экспорта cookies в файл `cookies.txt`
   - Скопируйте файл `cookies.txt` в корневую папку проекта

3. **Проверьте работу:**
   - Запустите программу - она автоматически использует cookies

### Решение 2: Использование PO Token (Альтернатива)

Код уже настроен для использования PO Token с pytubefix. Это может работать без дополнительных настроек.

### Решение 3: Ручная настройка yt-dlp

Если cookies не работают, можно использовать другие параметры аутентификации:

```bash
# Использование cookies из браузера
yt-dlp --cookies-from-browser chrome "URL"

# Или экспорт cookies
yt-dlp --cookies cookies.txt "URL"
```

## Что было исправлено в коде

1. **yt-dlp улучшения:**
   - Добавлен User-Agent для имитации браузера
   - Добавлена поддержка geo-bypass
   - Добавлена автоматическая загрузка cookies из файла `cookies.txt`

2. **pytubefix улучшения:**
   - Добавлен `use_po_token=True` для обхода блокировок
   - Добавлен дополнительный фолбэк с WEB клиентом

3. **Улучшенная обработка ошибок:**
   - Более информативные сообщения об ошибках
   - Автоматический переключение между методами

## Тестирование

Попробуйте загрузить видео с YouTube после применения одного из решений:

```bash
python Components/YoutubeDownloader.py
# Введите URL: https://www.youtube.com/watch?v=WKe8DvzOhV0
```

Если проблема сохраняется, проверьте:
- Наличие файла `cookies.txt` в корне проекта
- Правильность формата cookies файла
- Доступность интернета и YouTube

## Дополнительная информация

- [Документация yt-dlp по cookies](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp)
- [Документация pytubefix по PO Token](https://github.com/JuanBindez/pytubefix/pull/209)