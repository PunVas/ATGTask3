# import os
# import tempfile
# import gradio as gr
# import requests
# import json
# import time
# import torch
# import librosa
# import numpy as np
# import soundfile as sf
# from PIL import Image
# import base64
# from io import BytesIO
# import whisper
# from TTS.api import TTS

# # Configuration
# OLLAMA_API_BASE = "http://localhost:11434/api"
# WHISPER_MODEL = "base"
# OLLAMA_MODEL = "llava:latest"

# # Initialize models
# print("Loading Whisper model for speech recognition...")
# speech_recognizer = whisper.load_model(WHISPER_MODEL)

# print("Loading TTS model...")
# # Using CoquiTTS for high-quality speech output
# tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

# # Check if Ollama is running and has the required model
# def check_ollama_model():
#     try:
#         response = requests.get(f"{OLLAMA_API_BASE}/tags")
#         if response.status_code == 200:
#             models = response.json().get("models", [])
#             model_exists = any(model["name"] == OLLAMA_MODEL for model in models)
            
#             if not model_exists:
#                 print(f"Model {OLLAMA_MODEL} not found in Ollama. Please run 'ollama pull {OLLAMA_MODEL}'")
#                 return False
#             return True
#         else:
#             print("Failed to connect to Ollama API")
#             return False
#     except requests.exceptions.ConnectionError:
#         print("Could not connect to Ollama. Is it running?")
#         return False

# # Function to transcribe speech to text
# def transcribe_audio(audio_path):
#     try:
#         # Load audio
#         audio = whisper.load_audio(audio_path)
#         # Pad/trim audio to fit 30 seconds
#         audio = whisper.pad_or_trim(audio)
#         # Make log-Mel spectrogram
#         mel = whisper.log_mel_spectrogram(audio).to(speech_recognizer.device)
#         # Detect language
#         _, probs = speech_recognizer.detect_language(mel)
#         detected_language = max(probs, key=probs.get)
#         # Decode audio
#         options = whisper.DecodingOptions(fp16=torch.cuda.is_available())
#         result = whisper.decode(speech_recognizer, mel, options)
#         return result.text, detected_language
#     except Exception as e:
#         print(f"Error in transcription: {e}")
#         return "Error transcribing audio", "en"

# # Function to synthesize text to speech
# def synthesize_speech(text, output_path):
#     try:
#         tts_model.tts_to_file(text=text, file_path=output_path)
#         return output_path
#     except Exception as e:
#         print(f"Error in speech synthesis: {e}")
#         return None

# # Function to encode image to base64
# def encode_image_base64(image_path):
#     with open(image_path, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
#     return encoded_string

# # Function to generate response using Ollama's LLaVA model
# def generate_ollama_response(prompt, image_path=None):
#     try:
#         headers = {"Content-Type": "application/json"}
#         payload = {
#             "model": OLLAMA_MODEL,
#             "stream": False,
#             "options": {
#                 "temperature": 0.7,
#                 "num_predict": 1024,
#             }
#         }
        
#         if image_path:
#             # For LLaVA, we need to include the image as base64
#             image_base64 = encode_image_base64(image_path)
#             payload["prompt"] = prompt
#             payload["images"] = [image_base64]
#         else:
#             payload["prompt"] = prompt
        
#         response = requests.post(f"{OLLAMA_API_BASE}/generate", headers=headers, json=payload)
        
#         if response.status_code == 200:
#             return response.json().get("response", "No response received")
#         else:
#             error_msg = f"Error: {response.status_code} - {response.text}"
#             print(error_msg)
#             return error_msg
#     except Exception as e:
#         error_msg = f"Exception when calling Ollama API: {e}"
#         print(error_msg)
#         return error_msg

# EDUCATIONAL_CONTEXT = """
# You are an Educational AI Assistant designed to help students learn effectively.
# Your goal is to:
# 1. Provide clear, accurate answers to questions
# 2. Explain concepts in an accessible way appropriate for the student's level
# 3. Give helpful examples and analogies
# 4. Encourage critical thinking
# 5. Break down complex topics into simpler parts

# If you receive an image, analyze it carefully to understand any educational content,
# such as diagrams, equations, homework problems, or educational materials.

# When responding to voice inputs, be conversational but informative.
# """

# def process_input(text_input="", audio_input=None, image_input=None):
#     responses = []
#     audio_output_path = None
    
#     if audio_input is not None:
#         text_from_speech, detected_lang = transcribe_audio(audio_input)
#         responses.append(f"You said: {text_from_speech}")
#         text_input = text_from_speech
    
#     full_prompt = f"{EDUCATIONAL_CONTEXT}\n\nStudent query: {text_input}"
    
#     if image_input is not None:
#         ai_response = generate_ollama_response(full_prompt, image_input)
#         responses.append(f"Analyzing image and responding to: {text_input}\n\n{ai_response}")
#     else:
#         ai_response = generate_ollama_response(full_prompt)
#         responses.append(ai_response)
    
#     if ai_response:
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
#             audio_output_path = temp_audio.name
#             tts_text = ai_response[:500] + ("..." if len(ai_response) > 500 else "")
#             synthesize_speech(tts_text, audio_output_path)
    
#     return "\n\n".join(responses), audio_output_path

# def build_interface():
#     with gr.Blocks(title="Educational Multimodal AI Assistant") as app:
#         with gr.Row():
#             with gr.Column(scale=2):
#                 text_input = gr.Textbox(label="Type your question here",placeholder="Example: Explain Chmsky Normal Form in simple terms",lines=2)
#                 audio_input = gr.Audio(label="Or speak your question",type="filepath",sources="microphone")
#                 image_input = gr.Image(label="Upload an image (optional)",type="filepath")
#                 submit_btn = gr.Button("Get Answer", variant="primary")
            
#             with gr.Column(scale=3):
#                 output_text = gr.Textbox(label="Response",lines=15)
#                 output_audio = gr.Audio(label="Listen to the answer",type="filepath")
        
#         submit_btn.click(
#             fn=process_input,
#             inputs=[text_input, audio_input, image_input],
#             outputs=[output_text, output_audio]
#         )
    
#     return app

# if __name__ == "__main__":
  
#     app = build_interface()
#     app.launch(server_name="0.0.0.0", share=False, pwa=True)

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
