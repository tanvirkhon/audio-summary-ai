import openai
import os
from pydub import AudioSegment

openai.api_key = os.getenv("OPENAI_API_KEY")

# Transcribe audio Function
def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        transcription = openai.Audio.transcribe("whisper", audio_file)

    # Save the transcription to a text file
    with open("transcription.txt", "w") as f:
        f.write(transcription["text"])
        
    return transcription["text"]

# Abstract Summary Function
def abstract_summary_extraction(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points.",
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response["choices"][0]["message"]["content"]
    return transcription["text"]

# Key Points Extraction Function
def key_points_extraction(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information in key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about.",
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response["choices"][0]["message"]["content"]