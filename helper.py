import os
import librosa
import soundfile as sf
import pandas as pd
from datasets import Dataset, Audio

def process_audio_folder(data_folder, output_dir, max_duration=30):
    os.makedirs(output_dir, exist_ok=True)
    dataset = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".wav"):
            audio_path = os.path.join(data_folder, filename)
            base_filename = os.path.splitext(filename)[0]  
            transcription_path = os.path.join(data_folder, base_filename + ".txt")

            if os.path.exists(transcription_path):
                segments = process_audio_file(audio_path, transcription_path, output_dir, max_duration)
                dataset.extend(segments)

    return dataset

def process_audio_file(audio_path, transcription_path, output_dir, max_duration=30):
    sr = 16000  # Sampling rate in Hz
    full_audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    with open(transcription_path, 'r') as file:
        transcription_text = file.read()

    duration = librosa.get_duration(y=full_audio, sr=sr)
    num_segments = int(duration / max_duration) + 1

    segments = []
    for i in range(num_segments):
        start_time = i * max_duration
        end_time = min((i + 1) * max_duration, duration)
        segment_audio = full_audio[int(start_time * sr):int(end_time * sr)]

        segment_text = transcription_text[:segment_audio.shape[0]]  
        transcription_text = transcription_text[segment_audio.shape[0]:]

        segments.append({
            "audio": segment_audio,
            "text": segment_text,
            "start_time": start_time,
            "end_time": end_time
        })

    return segments

def create_dataset(data_folder, save_path):
    dataset = process_audio_folder(data_folder, save_path)
    audio_save_dir = os.path.join(save_path, "audio")
    os.makedirs(audio_save_dir, exist_ok=True)

    dataset_data = []
    for idx, item in enumerate(dataset):
        audio_filename = f"segment_{idx + 1}.wav"
        audio_dst_path = os.path.join(audio_save_dir, audio_filename)
        sf.write(audio_dst_path, item["audio"], 16000, subtype='PCM_16')  
        item["audio"] = audio_dst_path  
        dataset_data.append(item)

    train_data_dict = {
        "audio": [item["audio"] for item in dataset_data],
        "text": [item["text"] for item in dataset_data],
        "start_time": [item["start_time"] for item in dataset_data],
        "end_time": [item["end_time"] for item in dataset_data],
    }
    dataset = Dataset.from_dict(train_data_dict)
    dataset.save_to_disk(save_path)
    return dataset


data_folder = "./chich_speech_audio_files"
save_path = "./data/chichewa-dataset"

dataset = create_dataset(data_folder, save_path)
dataset.push_to_hub(repo_id="Rhode01/chichewa-dataset")
