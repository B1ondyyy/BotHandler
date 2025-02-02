from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.io as pio
import plotly.graph_objects as go
from PyQt5.QtWidgets import QDesktopWidget, QPlainTextEdit, QSizePolicy
from PyQt5.QtCore import QUrl, QObject
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog, QVBoxLayout, QPushButton, QProgressBar, \
    QLabel
from PyQt5.uic import loadUi

from collections import Counter
import whisper
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
import re
import torchaudio
from pathlib import Path
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from nemo.collections.asr.models import EncDecRNNTModel
import shutil
import tempfile
from pydub import AudioSegment
from pydub.effects import split_on_silence
import sys
import librosa
import numpy as np
import os
import noisereduce as nr
import torch
import soundfile as sf
from plotly.subplots import make_subplots

# Параметры настройки необходимые константы
AUDIO_SAMPLE_RATE = 16000  # Частота дискретизации аудиофайла
N_FFT = 1024  # Размер окна Фурье
HOP_LENGTH = 512  # Шаг окна Фурье
NO_OVERLAP = 512  # Перекрытие окон для визуализации спектрограммы

# Частотные диапазоны для гудков
BEEP_FREQUENCY_RANGES = [
    (410, 470),  # Диапазон для 420 Hz ± 30 Hz
    (480, 520),  # Диапазон для 500 Hz ± 20 Hz
]
MIN_BEEP_DURATION = 0.35  # Минимальная продолжительность гудка (в секундах)
SILENCE_PADDING = 0.05  # Добавочное время в начале и конце сегмента (в секундах)
FREQ_CHANGE_THRESHOLD = 75  # Максимально допустимое изменение частоты между соседними кадрами
MAX_CHAOTIC_FRAMES = 7

# Подключение CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Загрузка модели NVIDIA_TitaNet
model = EncDecRNNTModel.from_pretrained(model_name="nvidia/speakerverification_en_titanet_large")
model.to(device)

# Загрузка модели Silero VAD
silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
silero_model.to(device)
(get_speech_timestamps, _, read_audio, _, _) = utils

AMPLITUDE_THRESHOLD = 0.01
VAD_THRESHOLD = 0.85
MIN_SPEECH_DURATION = 0.3
input_folder = './tmp'
output_vad = './tmp_vad'
output_folder = './clustered audio'
output_segment = "./segments audio"
silence_folder = os.path.join(output_folder, "Тишина")
noise_folder = os.path.join(output_folder, "Шум")

Path(output_folder).mkdir(parents=True, exist_ok=True)
Path(silence_folder).mkdir(parents=True, exist_ok=True)
Path(noise_folder).mkdir(parents=True, exist_ok=True)


def process_audio_files(input_folder, output_folder):
    print("Выполняем предобработку аудиофайлов...")

    # Массив для хранения связи оригинальных файлов с обработанными
    original_to_vad = []
    vad_files = []  # Массив для хранения путей обработанных VAD файлов

    # Счетчики для тишины и шумов
    count_silence = 0
    count_noise = 0

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            input_path = os.path.join(input_folder, file_name)
            wav = read_audio(input_path)
            wav = wav.to(device)

            avg_amplitude = torch.mean(torch.abs(wav)).item()

            speech_timestamps = get_speech_timestamps(
                wav, silero_model, sampling_rate=16000, threshold=VAD_THRESHOLD
            )
            processed_audio = []


            for segment in speech_timestamps:
                segment_duration = (segment['end'] - segment['start']) / 16000
                if segment_duration >= MIN_SPEECH_DURATION:
                    processed_audio.append(wav[segment['start']:segment['end']])

            if avg_amplitude < AMPLITUDE_THRESHOLD:
                silence_path = os.path.join(silence_folder, file_name)
                torchaudio.save(silence_path, wav.unsqueeze(0).cpu(), sample_rate=16000)
                print(f"Тишина обнаружена и сохранена: {silence_path}")
                count_silence += 1
            elif not processed_audio:
                noise_path = os.path.join(noise_folder, file_name)
                torchaudio.save(noise_path, wav.unsqueeze(0).cpu(), sample_rate=16000)
                print(f"Шум обнаружен и сохранён: {noise_path}")
                count_noise += 1
            else:
                output_path = os.path.join(output_folder, f"{Path(file_name).stem}.wav")
                combined_audio = torch.cat(processed_audio, dim=0)
                torchaudio.save(output_path, combined_audio.cpu().unsqueeze(0), sample_rate=16000)
                print(f"Обработан и сохранён: {output_path}")

                # Добавляем в массив связь оригинального файла с обработанным
                original_to_vad.append((input_path, output_path))
                vad_files.append(output_path)  # Добавляем путь обработанного файла

    # Возвращаем два массива и счетчики: связи и пути обработанных файлов, количество тишины и шума
    return original_to_vad, vad_files, count_silence, count_noise


def load_audio(file_path, silence_threshold=AMPLITUDE_THRESHOLD):
    file_path = Path(file_path)
    if not file_path.is_file():
        print(f"Файл не найден или путь неверен: {file_path}")
        return None, None, True

    try:
        audio, sr = torchaudio.load(file_path)
        audio = audio.squeeze().numpy()

        if np.max(np.abs(audio)) < silence_threshold:
            print(f"Тишина обнаружена в файле: {file_path}")
            return None, sr, True

        return audio, sr, False

    except RuntimeError as e:
        print(f"Ошибка при загрузке аудиофайла: {file_path}, ошибка: {e}")
        return None, None, True


def get_audio_file_paths(folder):
    audio_extensions = ('.wav', '.mp3', '.flac')
    return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(audio_extensions)]


def get_embeddings(file_path):
    audio, sr, is_silence = load_audio(file_path)
    if is_silence or audio is None:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        temp_file_name = tmp_file.name
        torchaudio.save(temp_file_name, torch.tensor(audio).unsqueeze(0), sr)

    embeddings = model.get_embedding(path2audio_file=temp_file_name).cpu().detach().numpy().flatten()
    os.remove(temp_file_name)

    return embeddings


def get_all_embeddings(folder):
    # Получение списка путей к файлам
    original_file_paths = get_audio_file_paths(folder)

    # Загрузка DataFrame с метками
    df_tmp = pd.read_excel("Разметка 23-25.08.xlsx")

    file_paths = []
    embeddings_list = []
    real_classes = []
    embedding_size = None

    print("\nКластеризация...")

    for file_path in original_file_paths:
        embeddings = get_embeddings(file_path)

        if embeddings is not None:
            if embedding_size is None:
                embedding_size = embeddings.shape[0]

            if embeddings.shape[0] == embedding_size:
                embeddings_list.append(embeddings)
                file_paths.append(file_path)
                file_name = os.path.basename(file_path)

                try:
                    class_from_xlsx = df_tmp[df_tmp["Название файла"] == file_name]["Класс"].iloc[0]
                except IndexError:
                    class_from_xlsx = 'Нет информации'
                    print(f'Кластеризую файл: {file_name}')

                real_classes.append(class_from_xlsx)
            else:
                print(f"Пропуск файла {file_path} из-за некорректного размера эмбеддинга.")
        else:
            print(f"Эмбеддинг не извлечен для {file_path}")

    print(f"Эмбеддинги: {len(embeddings_list)}")
    print(f"Метки: {len(real_classes)}")

    return np.array(embeddings_list), file_paths, real_classes


def detect_beeps_spectral(audio_path, file_name, folder_name):
    # Загрузка аудиофайла
    y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)

    # Определяем метод анализа
    if folder_name.lower() in ["шум", "тишина"]:
        check = 0  # Хаотичные изменения
    else:
        check = 1  # Silero

    # Если Silero
    if check == 1:
        # Использование Silero VAD для поиска первого речевого сегмента
        audio_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        speech_timestamps = get_speech_timestamps(audio_tensor, silero_model, sampling_rate=AUDIO_SAMPLE_RATE)

        if speech_timestamps:
            first_speech_start = speech_timestamps[0]['start'] / AUDIO_SAMPLE_RATE
            print(f"Первый речевой сегмент начинается на {first_speech_start:.2f} секунде.")
        else:
            check = 0

    noise_sample = y[:int(sr * 0.5)]  # Используем первые 0.5 секунды как профиль шума
    y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)

    # Преобразование в спектр с использованием STFT
    D = librosa.stft(y_denoised, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude, _ = librosa.magphase(D)

    # Расчёт частоты и времена
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=HOP_LENGTH)

    # Анализ доминирующих частот оригинального сигнала
    D_original = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude_original, _ = librosa.magphase(D_original)
    dominant_frequencies_original = freq_bins[np.argmax(magnitude_original, axis=0)]

    beep_segments = []
    start_time = None
    previous_freq = None
    chaotic_count = 0

    for i, frame in enumerate(magnitude.T):
        current_time = times[i]
        if check == 1 and current_time >= first_speech_start:  # Прерываем анализ после начала речи
            break

        dominant_freq_idx = np.argmax(frame)
        dominant_freq = freq_bins[dominant_freq_idx]

        if check == 0:  # Проверка хаотичного изменения частоты на оригинальном сигнале
            dominant_freq_original = dominant_frequencies_original[i]  # Частота из оригинального сигнала
            if previous_freq is not None and abs(dominant_freq_original - previous_freq) > FREQ_CHANGE_THRESHOLD:
                chaotic_count += 1
            else:
                chaotic_count = 0
            previous_freq = dominant_freq_original

            if chaotic_count >= MAX_CHAOTIC_FRAMES:
                break

        # Проверка является ли текущая частота гудком
        is_beep = any(low <= dominant_freq <= high for low, high in BEEP_FREQUENCY_RANGES)

        if is_beep:
            if start_time is None:
                start_time = current_time
        else:
            if start_time is not None:
                end_time = current_time
                duration = end_time - start_time

                if duration >= MIN_BEEP_DURATION:
                    # Провка обнуления частоты в оригинальном сигнале в интервале 0.001–0.1 сек
                    start_idx = np.searchsorted(times, end_time)
                    end_idx = np.searchsorted(times, end_time + 0.4)
                    if end_idx > len(dominant_frequencies_original):
                        end_idx = len(dominant_frequencies_original)

                    if any(freq < 1 for freq in dominant_frequencies_original[start_idx:end_idx]):
                        beep_segments.append((max(0, start_time - SILENCE_PADDING),
                                              min(times[-1], end_time + SILENCE_PADDING)))
                start_time = None

    return beep_segments


def cut_beep(output_folder):
    for root, dirs, files in os.walk(output_folder):
        folder_name = os.path.basename(root)  # Название текущей папки

        # Игнор папки с названием "Без гудков"
        if folder_name.lower() == "Без гудков":
            continue

        for file_name in files:
            if file_name.lower().endswith(('.wav', '.mp3', '.flac')):  # Фильтруем аудиофайлы
                input_path = os.path.join(root, file_name)  # Полный путь к файлу
                beep_segments = detect_beeps_spectral(input_path, file_name, folder_name)  # Ваш метод обработки
                print(f"Папка '{folder_name}', файл '{file_name}' - обнаруженные гудки: {beep_segments}")

                # Путь для сохранения обработанных файлов
                no_beep_folder = os.path.join(root, 'Без гудков')
                os.makedirs(no_beep_folder, exist_ok=True)

                if beep_segments:  # Если гудки найдены
                    # Последний сегмент
                    _, last_end = beep_segments[-1]

                    # Загрузка аудио для обрезки
                    y, sr = librosa.load(input_path, sr=AUDIO_SAMPLE_RATE)

                    # Расчёт количество сэмплов для последнего времени
                    start_sample = int(last_end * sr)

                    # Удаление всего до окончания последнего гудка
                    y_trimmed = y[start_sample:]

                    # Сохранение обрезанного файла в папку Без гудков
                    trimmed_path = os.path.join(no_beep_folder, file_name)
                    sf.write(trimmed_path, y_trimmed, sr)

                    print(f"Файл {file_name} сохранен с {last_end:.2f} секунд до конца в {trimmed_path}")
                else:
                    # Если гудков нет, сохраняем оригинальный файл в папку "Без гудков"
                    original_path = os.path.join(no_beep_folder, file_name)
                    sf.write(original_path, librosa.load(input_path, sr=AUDIO_SAMPLE_RATE)[0], AUDIO_SAMPLE_RATE)
                    print(f"Файл {file_name} не содержит гудков. Оригинал сохранен в {original_path}.")


def cluster_audio_files_dbscan(embeddings, eps=0.17, min_samples=1):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings)
    return labels


def save_clusters_to_csv(file_paths, labels, real_classes, output_csv="clusters.csv"):
    df = pd.DataFrame({"File Path": file_paths, "Cluster": labels, "Real": real_classes})
    df = df.sort_values(by="Real", ascending=False)
    df.to_csv(output_csv, index=False)
    print(f"CSV файл с отношением аудиозаписей к кластерам сохранён как {output_csv}")


def copy_files_to_cluster_folders(file_paths, labels, output_folder="clustered audio"):
    cluster_dirs = {}

    for file_path, label in zip(file_paths, labels):
        if label not in cluster_dirs:
            cluster_folder = os.path.join(output_folder, f"Cluster {label}")
            os.makedirs(cluster_folder, exist_ok=True)
            cluster_dirs[label] = cluster_folder
        shutil.copy(file_path, cluster_dirs[label])

    print(f"Все файлы успешно скопированы по кластерам в папку '{output_folder}'")


def print_clustering_metrics(labels, embeddings):
    if len(labels) != len(embeddings):
        print(f"Ошибка: количество меток ({len(labels)}) не совпадает с количеством эмбеддингов ({len(embeddings)})")
        return

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"Количество кластеров: {n_clusters}")
    print(f"Количество выбросов: {n_noise}")

    if n_clusters > 1:
        silhouette_avg = silhouette_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        print(f"Средний силуэтный коэффициент: {silhouette_avg:.4f}")
        print(f"Индекс Дейвиса-Боулдина: {davies_bouldin:.4f}")
    else:
        print("Недостаточно кластеров для расчета силуэтного коэффициента и индекса Дейвиса-Боулдина.")

    return n_clusters, n_noise

def plot_3d_clusters_plotly(embeddings, labels, file_paths, n_clusters, n_noise, count_silence, count_noise):
    # Создание субплотов
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{'type': 'scatter3d'}],
               [{'type': 'bar'}]],
        subplot_titles=('', 'Статистика кластеров'),
        row_heights=[0.7, 0.3],  # 70% для 3D графика, 30% для столбчатой диаграммы
        vertical_spacing=0.05  # Отступ между субплотами
    )

    # 3D Кластерный график
    embeddings_3d = PCA(n_components=3).fit_transform(embeddings)
    unique_labels = np.unique(labels)
    marker_shapes = ['circle', 'square', 'diamond', 'cross', 'x']

    for label in unique_labels:
        cluster_points = embeddings_3d[labels == label]
        cluster_file_paths = np.array(file_paths)[labels == label]
        marker_shape = marker_shapes[label % len(marker_shapes)] if label != -1 else 'circle'  # Форма маркера для выбросов

        if label == -1:
            color = 'red'
            name = 'Выбросы'
        else:
            name = f'Кластер {label}'

        trace = go.Scatter3d(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            z=cluster_points[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=label,
                opacity=0.8,
                symbol=marker_shape
            ),
            text=cluster_file_paths,
            hoverinfo='text',
            name=name
        )
        fig.add_trace(trace, row=1, col=1)

    # Диаграмма столбцов
    categories = ['Выбросы', 'Кластеры', 'Шум', 'Тишина']
    counts = [n_noise, n_clusters, count_noise, count_silence]

    bar_trace = go.Bar(
        x=categories,
        y=counts,
        marker=dict(color=['blue', 'orange', 'green', 'gray']),
        text=counts,
        textposition='auto'
    )
    fig.add_trace(bar_trace, row=2, col=1)

    # Настройка осей для столбчатой диаграммы
    fig.update_yaxes(title_text='Количество', row=2, col=1)
    fig.update_xaxes(title_text='Категории', row=2, col=1)

    # Оформление
    fig.update_layout(
        title='Кластеризация аудиофайлов и статистика',
        showlegend=True,
        height=1200,
        margin=dict(l=50, r=50, t=100, b=50)
    )

    # Настройка камеры 3D графика для лучшего обзора
    fig.update_scenes(
        aspectmode='cube',
        camera=dict(
            eye=dict(x=2, y=2, z=2)
        )
    )

    return fig


def save_clusters_with_originals(file_paths, labels, original_to_vad, output_folder="clustered audio"):
    # Создаем маппинг из обработанных файлов в оригиналы
    vad_to_original = {vad: original for original, vad in original_to_vad}
    os.makedirs(output_folder, exist_ok=True)

    for file_path, label in zip(file_paths, labels):
        # Получаем оригинальный файл
        original_path = vad_to_original[file_path]

        # Определение папки для текущего кластера
        if label == -1:
            cluster_folder = os.path.join(output_folder, "Noise")
        else:
            cluster_folder = os.path.join(output_folder, f"Cluster_{label}")

        os.makedirs(cluster_folder, exist_ok=True)

        # Копирование оригинального файла в соответствующий кластер
        shutil.copy(original_path, cluster_folder)

    print(f"Оригиналы сохранены в папках кластеров: {output_folder}")


# Загрузка русского стоп-словаря
nltk.download("stopwords")
stop_words = set(stopwords.words("russian"))
trans_model = whisper.load_model("large")
trans_model.to(device)


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


def filter_stopwords(text):
    text = text.lower()
    text = remove_punctuation(text)  # Удаляем знаки препинания
    words = [word for word in text.split() if word not in stop_words or word == "не"]
    return " ".join(words)


# Функция для нахождения наиболее частой фразы
def find_frequent_phrase(text, min_len, max_len):
    words = text.split()
    candidates = []
    for n in range(min_len, max_len + 1):
        candidates.extend(ngrams(words, n))
    phrases = [" ".join(ngram) for ngram in candidates]
    phrase_counts = Counter(phrases)
    most_common_phrase = phrase_counts.most_common(1)
    return most_common_phrase[0][0] if most_common_phrase else None


def remove_unwanted_phrases(text):
    unwanted_phrases = ["Продолжение следует...", "продолжение следует"]
    for phrase in unwanted_phrases:
        text = text.replace(phrase, "")
    return text


def truncate_to_words(text, max_words):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text

def sanitize_filename(filename):
    """
    Удаляем недопустимые символы из имени файла
    """
    invalid_chars = r'[<>:"/\\|?*]'
    # Замена недопустимых символов на пустую строку
    sanitized = re.sub(invalid_chars, '', filename)
    sanitized = sanitized.strip()
    return sanitized

# Функция для переименования папки
def rename_cluster_folder(parent_folder, old_name, new_name):
    old_path = os.path.join(parent_folder, old_name)
    new_path = os.path.join(parent_folder, new_name)

    counter = 1
    while os.path.exists(new_path):
        new_path = os.path.join(parent_folder, f"{new_name} ({counter})")
        counter += 1

    os.rename(old_path, new_path)
    print(f"   Папка переименована: {old_name} -> {os.path.basename(new_path)}")

# Функция для обработки кластеров
def process_clusters(folder_path):
    for cluster_folder in os.listdir(folder_path):
        cluster_path = os.path.join(folder_path, cluster_folder)

        if "-" in cluster_folder:  # Добавление проверки на суффикс после "-"
            print(f"Пропуск кластера: {cluster_folder} (папка уже переименована)")
            continue

        if not os.path.isdir(cluster_path):
            continue

        # Пропуск папки с именами "шум" или "тишина"
        if cluster_folder.lower() in {"шум", "тишина"}:
            print(f"Пропуск кластера: {cluster_folder} (шум или тишина)")
            continue

        print(f"Обработка кластера: {cluster_folder}")
        audio_files = [f for f in os.listdir(cluster_path) if f.endswith((".mp3", ".wav"))]

        # Если в кластере нет файлов, пропуск
        if not audio_files:
            print(f"   Пропуск кластера {cluster_folder}, так как он пустой.")
            continue

        # Если в кластере только один файл, обрабатываем его отдельно
        if len(audio_files) == 1:
            single_audio_file = audio_files[0]
            audio_path = os.path.join(cluster_path, single_audio_file)
            print(f"   Транскрибируется единственный файл: {single_audio_file}")
            result = trans_model.transcribe(audio_path, language="ru")
            transcribed_text = result["text"]

            # Удаление нежелательных фраз
            cluster_title = remove_unwanted_phrases(transcribed_text)
            final_cluster_title = truncate_to_words(f"{cluster_folder} - {cluster_title}", 9)
            print(f"   Название для кластера (1 файл): {final_cluster_title}")

            sanitized_title = sanitize_filename(final_cluster_title)

            # Сохранение результатов
            with open(os.path.join(cluster_path, "cluster_text.txt"), "w", encoding="utf-8") as summary_file:
                summary_file.write(f"Название кластера: {final_cluster_title}\n")
                summary_file.write("Полный текст:\n")
                summary_file.write(transcribed_text)

            # Переименовывание папки
            rename_cluster_folder(folder_path, cluster_folder, sanitized_title)
            continue  # Переход к следующему кластеру

        # Для кластеров с несколькими файлами
        all_text = []
        total_word_count = 0

        for audio_file in audio_files:
            audio_path = os.path.join(cluster_path, audio_file)
            print(f"   Транскрибируется файл: {audio_file}")
            result = trans_model.transcribe(audio_path, language="ru")
            transcribed_text = result["text"]

            # Удаление стоп-слов
            filtered_text = filter_stopwords(transcribed_text)
            all_text.append(filtered_text)

            # Подсчёт слов
            total_word_count += len(filtered_text.split())

        # Объединение текста всех файлов кластера
        combined_text = " ".join(all_text)
        combined_text = remove_unwanted_phrases(combined_text)

        # Вычисление среднего количество слов
        avg_word_count = total_word_count // len(audio_files)

        # Динамический диапазон для длины фраз
        if avg_word_count <= 4:
            min_len = 1
            max_len = 4
        elif avg_word_count <= 10:
            min_len = 5
            max_len = 6
        else:
            min_len = 7
            max_len = 9

        # Поиск наиболее частой фразы
        cluster_title = find_frequent_phrase(combined_text, min_len, max_len)
        final_cluster_title = f"{cluster_folder} - {cluster_title}"
        print(f"   Название для кластера: {final_cluster_title} (мин: {min_len}, макс: {max_len})")

        sanitized_title = sanitize_filename(final_cluster_title)

        # Сохранение результата
        with open(os.path.join(cluster_path, "cluster_text.txt"), "w", encoding="utf-8") as summary_file:
            summary_file.write(f"Название кластера: {final_cluster_title}\n")
            summary_file.write("Полный текст:\n")
            summary_file.write(combined_text)

        # Переименование папки
        rename_cluster_folder(folder_path, cluster_folder, sanitized_title)


# Путь к папке с кластерами
clusters_folder = "./clustered audio"


def split_audio_on_silence_with_dynamic_thresh(audio_segment, silence_padding=500):
    """Разделение аудиофайла на сегменты по тишине"""
    silence_thresh = audio_segment.dBFS - 16

    print(f"AVG audio volume: {audio_segment.dBFS} dB")
    print(f"Using silence trash: {silence_thresh} dB")
    segments = split_on_silence(audio_segment, min_silence_len=500,
                                silence_thresh=silence_thresh, keep_silence=silence_padding)
    print(f"Segments found: {len(segments)}")
    return segments


def apply_gain(audio_segment):
    """
    Усиление звука: регулируем громкость в зависимости от исходной максимальной амплитуды.
    """
    max_amplitude = audio_segment.max_dBFS
    gain_value = 2 if max_amplitude >= -2 else 4 if max_amplitude <= -4 else 3
    print(f"Applying gain: {gain_value} dB to the audio.")
    return audio_segment.apply_gain(gain_value)


def segment_audio_files(input_folder, output_folder):
    """Обработка аудиофайлов из input_folder и сохранение сегментов в output_folder"""
    # Создание папки для сегментов, если её ещё нет
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)

            # Создание отдельной папки для сегментов каждого файла
            output_file_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
            Path(output_file_folder).mkdir(parents=True, exist_ok=True)

            try:
                # Загрузка аудиофайла
                audio = AudioSegment.from_wav(input_path)

                # Усиление громкости
                audio = apply_gain(audio)

                # Обрезка по тишине и удаление гудков и коротких сегментов
                audio_segments = split_audio_on_silence_with_dynamic_thresh(audio)

                for i, segment in enumerate(audio_segments):
                    # Формирование имени файла для сохранения
                    output_filename = f"{os.path.splitext(filename)[0]}_{i + 1}.wav"
                    output_path = os.path.join(output_file_folder, output_filename)

                    # Сохранение сегмента
                    segment.export(output_path, format="wav")
                    print(f"Обработан файл: {output_path}")
            except Exception as e:
                print(f"Ошибка при обработке файла {input_path}: {e}")


folder_path = 'tmp_vad'
embeddings, file_paths, real_classes = get_all_embeddings(folder_path)


# Worker для выполнения всех операций в потоке
class Worker(QThread):
    progress_changed = pyqtSignal(int)
    status_changed = pyqtSignal(str)
    visualization_ready = pyqtSignal(str)  # Сигнал для передачи пути к HTML

    def __init__(self, folder, is_segmentation=False):
        super(Worker, self).__init__()
        self.folder = folder
        self.is_segmentation = is_segmentation

    def run(self):
        if self.is_segmentation:
            self.status_changed.emit("Сегментация началась...")
            segment_audio_files(self.folder, './segments audio')
            self.progress_changed.emit(100)
            self.status_changed.emit("Сегментация завершена!")
        else:
            self.status_changed.emit("Предобработка началась...")
            original_to_vad, vad_files, count_silence, count_noise = process_audio_files(self.folder, './tmp_vad')
            self.progress_changed.emit(25)

            self.status_changed.emit("Кластеризация началась...")
            folder_path = 'tmp_vad'
            embeddings, file_paths, real_classes = get_all_embeddings(folder_path)
            labels = cluster_audio_files_dbscan(embeddings, eps=0.17, min_samples=1)
            n_clusters, n_noise = print_clustering_metrics(labels, embeddings)
            save_clusters_to_csv(file_paths, labels, real_classes, output_csv="clusters.csv")
            self.progress_changed.emit(75)
            save_clusters_with_originals(vad_files, labels, original_to_vad, output_folder="clustered audio")
            self.status_changed.emit("Удаляю гудки из файлов...")
            cut_beep(output_folder="clustered audio")
            self.status_changed.emit("Слушаю аудиофайлы и даю имена кластерам...")
            process_clusters(clusters_folder)

            self.status_changed.emit("Визуализация началась...")
            self.progress_changed.emit(100)

            # Генерация комбинированного графика
            fig = plot_3d_clusters_plotly(
                embeddings,
                labels,
                file_paths,
                n_clusters=n_clusters,
                n_noise=n_noise,
                count_silence=count_silence,
                count_noise=count_noise
            )

            # Параметры конфигурации Plotly для отключения зума
            config = {
                'displayModeBar': True,  # Отображаем панель инструментов
                'modeBarButtonsToRemove': [
                    'zoom2d', 'zoomIn2d', 'zoomOut2d',  # Кнопки зума для 2D графиков
                    'zoom3d', 'zoomIn3d', 'zoomOut3d',  # Кнопки зума для 3D графиков
                    'resetScale2d', 'resetScale3d'  # Кнопки сброса масштаба
                ]
            }

            # Путь к HTML файлу
            file_path = os.path.join(os.getcwd(), 'cluster_visualization.html')
            # Сохранение графика в HTML файл
            pio.write_html(fig, file_path, config=config)
            # Отправка пути к HTML файлу для отображения
            self.visualization_ready.emit(file_path)  # Передача строки пути


class EmittingStream(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))

    def flush(self):
        pass  # Необходимо для совместимости с sys.stdout


# Основной интерфейс с PyQt
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        loadUi('main_window.ui', self)
        screen = QDesktopWidget().screenGeometry()
        window = self.frameGeometry()
        center_point = screen.center()
        window.moveCenter(center_point)
        self.move(window.topLeft())

        # Устанавливаем иконку окна
        icon_path = './app_icon.ico'
        self.setWindowIcon(QIcon(icon_path))
        self.select_folder_button.clicked.connect(self.select_folder)
        layout = QVBoxLayout()
        layout.addWidget(self.select_folder_button)
        layout.setAlignment(Qt.AlignCenter)
        central_widget = QWidget()
        central_widget.setLayout(layout)

        self.setCentralWidget(central_widget)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с аудиофайлами")
        if folder:
            self.open_action_window(folder)

    def open_action_window(self, folder):
        self.action_window = ActionWindow(folder)
        self.close()
        self.action_window.show()


class ActionWindow(QDialog):
    def __init__(self, folder):
        super(ActionWindow, self).__init__()
        self.setWindowTitle("Выберите действие")
        self.setGeometry(100, 100, 560, 250)

        screen = QDesktopWidget().screenGeometry()
        window = self.frameGeometry()
        center_point = screen.center()
        window.moveCenter(center_point)
        self.move(window.topLeft())

        icon_path = './app_icon.ico'
        self.setWindowIcon(QIcon(icon_path))

        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.folder = folder
        layout = QVBoxLayout()

        self.clustering_button = QPushButton("Кластеризация")
        self.clustering_button.clicked.connect(self.start_clustering)
        layout.addWidget(self.clustering_button)

        self.segmentation_button = QPushButton("Сегментация")
        self.segmentation_button.clicked.connect(self.start_segmentation)
        layout.addWidget(self.segmentation_button)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ожидание действия...", self)
        layout.addWidget(self.status_label)

        # Кнопка для показа/скрытия логов
        self.toggle_log_button = QPushButton("Показать подробности")
        self.toggle_log_button.setCheckable(True)
        self.toggle_log_button.clicked.connect(self.toggle_log)
        layout.addWidget(self.toggle_log_button)

        # Виджет для логов
        self.log_text = QPlainTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)  # Ограничение высоты

        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.log_text.hide()  # Скрыт по умолчанию
        layout.addWidget(self.log_text)

        self.setLayout(layout)

        # Установка перенаправления вывода
        self.stream = EmittingStream()
        self.stream.text_written.connect(self.append_text)
        sys.stdout = self.stream
        sys.stderr = self.stream

    def append_text(self, text):
        """Добавление текста в виджет логов без лишних отступов."""
        self.log_text.appendPlainText(text.rstrip())
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def toggle_log(self, checked):
        """Переключение видимости виджета логов."""
        if checked:
            self.log_text.show()
            self.toggle_log_button.setText("Скрыть подробности")
        else:
            self.log_text.hide()
            self.toggle_log_button.setText("Показать подробности")

    def start_clustering(self):
        self.status_label.setText("Предобработка, кластеризация начались...")
        self.progress_bar.setValue(0)

        # Отключаем кнопки
        self.clustering_button.setEnabled(False)
        self.segmentation_button.setEnabled(False)

        self.worker = Worker(self.folder, is_segmentation=False)
        self.worker.progress_changed.connect(self.update_progress)
        self.worker.status_changed.connect(self.update_status)
        self.worker.visualization_ready.connect(self.show_visualization)  # Подключение сигнала для визуализации
        self.worker.finished.connect(self.enable_buttons)  # Подключение сигнала завершения к разблокировке кнопок
        self.worker.start()

    def start_segmentation(self):
        self.status_label.setText("Сегментация...")
        self.progress_bar.setValue(0)

        # Отключение кнопки
        self.clustering_button.setEnabled(False)
        self.segmentation_button.setEnabled(False)

        self.worker = Worker(self.folder, is_segmentation=True)
        self.worker.progress_changed.connect(self.update_progress)
        self.worker.status_changed.connect(self.update_status)
        self.worker.finished.connect(self.enable_buttons)  # Подключение сигнала завершения к разблокировке кнопок
        self.worker.start()

    def enable_buttons(self):
        self.clustering_button.setEnabled(True)
        self.segmentation_button.setEnabled(True)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, status):
        self.status_label.setText(status)

    def show_visualization(self, html_path):
        """Открытие нового окна с визуализацией"""
        self.visualization_window = VisualizationWindow(html_path)  # Передача строки пути
        self.visualization_window.show()

    def restore_stdout(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


class VisualizationWindow(QMainWindow):
    def __init__(self, html_path):
        super().__init__()
        self.setWindowTitle("Визуализация кластеров")
        self.setGeometry(100, 100, 1420, 750)
        icon_path = './app_icon.ico'
        self.setWindowIcon(QIcon(icon_path))

        self.web_view = QWebEngineView(self)
        self.web_view.setUrl(QUrl.fromLocalFile(html_path))  # Преобразование строки пути в QUrl
        self.setCentralWidget(self.web_view)


# Запуск приложения
if __name__ == "__main__":
    app = QApplication(sys.argv)

    icon_path = './app_icon.ico'
    app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())