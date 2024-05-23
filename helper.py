
import os
import webvtt
import librosa
import soundfile as sf
from datasets import Dataset,  Audio, DatasetDict
from datetime import datetime
import torch



repo_name = "chichewa-dataset"
data_folder = "./chich_speech_audio_files"
save_path = f"data/{repo_name}-dataset"
hf_username = "Rhode01"
def parse_time(time_str):
    hours, minutes, seconds_milliseconds = time_str.split(':')
    seconds, milliseconds = seconds_milliseconds.split('.')
    
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    milliseconds = int(milliseconds)
    
 
    parsed_time = datetime(year=1900, month=1, day=1, hour=hours, minute=minutes, second=seconds, microsecond=milliseconds * 1000)
    
    return parsed_time

def time_to_samples(time_ms, sr):
    return int((time_ms / 1000.0) * sr)

def transform_data(data):
    transformed = {"audio": [], "text": [], "start_time": [], "end_time": []}
    for item in data:
        for key in transformed:
            transformed[key].append(item[key])
    return transformed
def process_audio_folder(data_folder, output_dir, max_duration=30):
    os.makedirs(output_dir, exist_ok=True)
    data = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".wav"):
            audio_path = os.path.join(data_folder, filename)
            base_filename = os.path.splitext(filename)[0]  
            transcription_path = os.path.join(data_folder, base_filename + ".txt")
            if os.path.exists(transcription_path):
                try:
                    data.extend(process_audio_file(audio_path, transcription_path, output_dir, max_duration))
                except OSError as e:
                    print(f"Error processing {audio_path} or {transcription_path}: {e}")
    return data

def process_audio_file(audio_path, transcription_path, output_dir, max_duration=30):
    full_audio, sr = librosa.load(audio_path, sr=None, mono=True)

    with open(transcription_path, 'r') as file:
        transcription_text = file.read()

    data = []
    current_text = []
    current_start = None
    accumulated_duration = 0
    segment_counter = 0

    for caption in webvtt.from_srt(transcription_text):
        start_time = parse_time(caption.start)
        end_time = parse_time(caption.end)
        duration = (end_time - start_time).total_seconds()

        if current_start is None:
            current_start = start_time

        if accumulated_duration + duration <= max_duration:
            current_text.append(caption.text)
            accumulated_duration += duration
        else:
            segment_counter += 1
            segment_filename = f"{output_dir}/segment_{segment_counter}.wav"
            start_sample = time_to_samples(current_start, sr)
            end_sample = time_to_samples(current_start + max_duration, sr)
            audio_segment = full_audio[start_sample:end_sample]
            sf.write(segment_filename, audio_segment, sr, format='wav')

            data.append({
                "audio": segment_filename,
                "text": " ".join(current_text),
                "start_time": current_start.strftime("%H:%M:%S.%f"),
                "end_time": (current_start + max_duration).strftime("%H:%M:%S.%f")
            })

            current_text = [caption.text]
            current_start = start_time
            accumulated_duration = duration

    if current_text:
        segment_counter += 1
        segment_filename = f"{output_dir}/segment_{segment_counter}.wav"
        start_sample = time_to_samples(current_start, sr)
        end_sample = len(full_audio)
        audio_segment = full_audio[start_sample:end_sample]
        sf.write(segment_filename, audio_segment, sr, format='wav')

        data.append({
            "audio": segment_filename,
            "text": " ".join(current_text),
            "start_time": current_start.strftime("%H:%M:%S.%f"),
            "end_time": end_time.strftime("%H:%M:%S.%f")
        })

    return data

def create_dataset(data_folder):
    train_data = process_audio_folder(data_folder, save_path)
    train_dataset = Dataset.from_dict(transform_data(train_data))
    dataset_dict = DatasetDict({
        "train": train_dataset,
    })
    return dataset_dict

dataset = create_dataset(data_folder)

dataset.save_to_disk(save_path)
dataset.cast_column("audio", Audio())
dataset.push_to_hub(repo_id=f"{hf_username}/{repo_name}")
