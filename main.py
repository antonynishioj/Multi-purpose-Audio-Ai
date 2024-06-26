import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import OneHotEncoder
from keras.models import load_model
import gradio as gr

import subprocess
import whisperx
import whisper
import gc
import gradio as gr
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from spleeter.separator import Separator
import IPython.display as ipd

# Code for Task 1
def predict_emotion(audio_file):
    model = load_model("/content/speechemotion.h5")
    y, sr = librosa.load(audio_file, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    # Extract MFCC features
    # Reshape MFCC features
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    # Make prediction
    prediction = model.predict(mfcc)
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction)
    # Map index to emotion class
    emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'calm', 'sad']
    predicted_emotion = emotion_classes[predicted_class_index]
    return predicted_emotion

# Code for Task 2
def transcribe_and_detect_language(audio_file_path):
    # Load the model
    model = whisper.load_model("base")
    
    # Load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_file_path)
    audio = whisper.pad_or_trim(audio)
    
    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    # Detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    
    # Decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    
    # Return the transcript and detected language
    return "Transcript \n"+result.text, "Language \n"+detected_language

#task 3
def MultiSpeakerRecognition(audio_input):
    device = "cuda"
    batch_size = 4 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    audio_file = audio_input
    audio = whisperx.load_audio(audio_file)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size)
#print(result["segments"]) # before alignment

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

#print(result)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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",
                                             device=device)
    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
#print(diarize_segments)
#print(diarize_segments.speaker.unique())

    result = whisperx.assign_word_speakers(diarize_segments, result)
    return str(diarize_segments),result["segments"]
#print(result["segments"]) # segments are now assigned speaker IDs

def split_audio(audio_file):
    # Define the path to your audio file
    audio_file = audio_file
    output1 = audio_file.split("/")[-1]
    paths = output1.split(".")[0]
    # Initialize the separator
    separator = Separator('spleeter:5stems')
    # Perform source separation
    separator.separate_to_file(audio_file, '/content/')
    vocals_audio_file = "/content/"+paths+"/vocals.wav"
    drums_audio_file = "/content/"+paths+"/drums.wav"
    bass_audio_file = "/content/"+paths+"/bass.wav"
    piano_audio_file = "/content/"+paths+"/piano.wav"
    other_audio_file = "/content/"+paths+"/other.wav"

    return vocals_audio_file,drums_audio_file,bass_audio_file,piano_audio_file,other_audio_file

# interface one
iface1 = gr.Interface(
    fn=predict_emotion,
    inputs=gr.components.Audio(type = 'filepath',label="Upload Audio File"),
    outputs="text",
    title="Emotion Detection"
)
# interface two
iface2 = gr.Interface(
    fn=transcribe_and_detect_language,
    inputs=gr.components.Audio(type = 'filepath',label="Upload Audio File"),
    outputs=["text","text"],
    title="Transcript and Language"
)
#interface three
iface3 = gr.Interface(
    fn=MultiSpeakerRecognition,
    inputs=gr.components.Audio(type = 'filepath',label="Upload Audio File"),
    outputs=["text","text"],
    title="Multiple Speaker"
)
#interface four
iface4 = gr.Interface(
    fn=split_audio,
    inputs=gr.components.Audio(type = 'filepath',label="Upload Audio File"),
    outputs=["audio","audio","audio","audio","audio"],
    title="Split Audio"
)


demo = gr.TabbedInterface([iface1, iface2, iface3, iface4], ["Emotion", "Transcript and Language","Multi Speaker","Split Audio"])

# Run the interface
demo.launch(share=True, debug=True)
