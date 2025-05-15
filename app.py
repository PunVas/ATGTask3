import os
import tempfile
import gradio as gr
import requests
import json
import time
import torch
import librosa
import numpy as np
import soundfile as sf
from PIL import Image
import base64
from io import BytesIO
import whisper
from TTS.api import TTS

OLLAMA_API_BASE = "http://localhost:11434/api"
WHISPER_MODEL = "base"
OLLAMA_MODEL = "llava:latest"

print("Starting up, loading models...")
speech_recognizer = whisper.load_model(WHISPER_MODEL)
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

def check_ollama_model():
    try:
        res = requests.get(f"{OLLAMA_API_BASE}/tags")
        if res.status_code == 200:
            tags = res.json().get("models", [])
            found = False
            for m in tags:
                if m.get("name") == OLLAMA_MODEL:
                    found = True
                    break
            if not found:
                print(f"{OLLAMA_MODEL} isn't loaded in Ollama.")
                return False
            return True
        print("Failed to connect to Ollama.")
        return False
    except requests.exceptions.ConnectionError:
        print("Ollama not running?")
        return False

def transcribe_audio(audio_fp):
    try:
        raw = whisper.load_audio(audio_fp)
        raw = whisper.pad_or_trim(raw)
        spec = whisper.log_mel_spectrogram(raw).to(speech_recognizer.device)
        _, lang_scores = speech_recognizer.detect_language(spec)
        lang = max(lang_scores, key=lang_scores.get)
        opts = whisper.DecodingOptions(fp16=torch.cuda.is_available())
        res = whisper.decode(speech_recognizer, spec, opts)
        return res.text, lang
    except Exception as err:
        print(f"Oops: {err}")
        return "Error transcribing audio", "en"

def synthesize_speech(text_data, out_fp):
    try:
        tts_model.tts_to_file(text=text_data, file_path=out_fp)
        return out_fp
    except Exception as err:
        print(f"Synthesis fail: {err}")
        return None

def encode_image_base64(img_path):
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded

def generate_ollama_response(text_prompt, img_path=None):
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1024
            }
        }
        if img_path:
            encoded_img = encode_image_base64(img_path)
            data["prompt"] = text_prompt
            data["images"] = [encoded_img]
        else:
            data["prompt"] = text_prompt
        r = requests.post(f"{OLLAMA_API_BASE}/generate", headers=headers, json=data)
        if r.status_code == 200:
            return r.json().get("response", "Empty reply")
        else:
            err = f"Error {r.status_code}: {r.text}"
            print(err)
            return err
    except Exception as err:
        print(f"Ollama call exception: {err}")
        return str(err)

EDUCATIONAL_CONTEXT = """
You are an Educational AI Assistant designed to help students learn effectively.
Your goal is to:
1. Provide clear, accurate answers to questions
2. Explain concepts in an accessible way appropriate for the student's level
3. Give helpful examples and analogies
4. Encourage critical thinking
5. Break down complex topics into simpler parts

If you receive an image, analyze it carefully to understand any educational content,
such as diagrams, equations, homework problems, or educational materials.

When responding to voice inputs, be conversational but informative.
"""

def process_input(text_input="", audio_input=None, image_input=None):
    result_msgs = []
    audio_out = None

    if audio_input:
        said_text, language = transcribe_audio(audio_input)
        result_msgs.append(f"You said: {said_text}")
        text_input = said_text

    complete_prompt = f"{EDUCATIONAL_CONTEXT}\n\nStudent query: {text_input}"

    if image_input:
        bot_answer = generate_ollama_response(complete_prompt, image_input)
        result_msgs.append(f"Analyzing image and responding to: {text_input}\n\n{bot_answer}")
    else:
        bot_answer = generate_ollama_response(complete_prompt)
        result_msgs.append(bot_answer)

    if bot_answer:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio_out = temp_file.name
            short_text = bot_answer[:500] + ("..." if len(bot_answer) > 500 else "")
            synthesize_speech(short_text, audio_out)

    return "\n\n".join(result_msgs), audio_out

def build_interface():
    with gr.Blocks(title="Educational Multimodal AI Assistant") as ui:
        with gr.Row():
            with gr.Column(scale=2):
                txt_in = gr.Textbox(label="Type your question here", placeholder="Example: Explain Chmsky Normal Form in simple terms", lines=2)
                mic_in = gr.Audio(label="Or speak your question", type="filepath", sources="microphone")
                img_in = gr.Image(label="Upload an image (optional)", type="filepath")
                btn_submit = gr.Button("Get Answer", variant="primary")
            with gr.Column(scale=3):
                txt_out = gr.Textbox(label="Response", lines=15)
                audio_out = gr.Audio(label="Listen to the answer", type="filepath")

        btn_submit.click(
            fn=process_input,
            inputs=[txt_in, mic_in, img_in],
            outputs=[txt_out, audio_out]
        )

    return ui

if __name__ == "__main__":
    app = build_interface()
    app.launch(server_name="0.0.0.0", share=False, pwa=True)
