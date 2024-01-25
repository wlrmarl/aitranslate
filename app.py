
import io
import subprocess
import soundfile
import torchaudio
import torch
import os

from pydub import AudioSegment
from flask import Flask, render_template, request, jsonify, send_file
from seamless_communication.inference import Translator
from seamless_communication.streaming.dataloaders.s2tt import SileroVADSilenceRemover



# List of target languages and their codes
target_lang = [
    {"code": "arb", "name": "Arabic"},
    {"code": "ben", "name": "Bengali"},
    {"code": "cat", "name": "Catalan"},
    {"code": "ces", "name": "Czech"},
    {"code": "cmn", "name": "Mandarin Chinese"},
    {"code": "cym", "name": "Welsh"},
    {"code": "dan", "name": "Danish"},
    {"code": "deu", "name": "German"},
    {"code": "eng", "name": "English"},
    {"code": "est", "name": "Estonian"},
    {"code": "fin", "name": "Finnish"},
    {"code": "fra", "name": "French"},
    {"code": "hin", "name": "Hindi"},
    {"code": "ind", "name": "Indonesian"},
    {"code": "ita", "name": "Italian"},
    {"code": "jpn", "name": "Japanese"},
    {"code": "kan", "name": "Kannada"},
    {"code": "kor", "name": "Korean"},
    {"code": "mlt", "name": "Maltese"},
    {"code": "nld", "name": "Dutch"},
    {"code": "pes", "name": "Western Persian"},
    {"code": "pol", "name": "Polish"},
    {"code": "por", "name": "Portuguese"},
    {"code": "ron", "name": "Romanian"},
    {"code": "rus", "name": "Russian"},
    {"code": "slk", "name": "Slovak"},
    {"code": "spa", "name": "Spanish"},
    {"code": "swe", "name": "Swedish"},
    {"code": "swh", "name": "Swahili"},
    {"code": "tam", "name": "Tamil"},
    {"code": "tel", "name": "Telugu"},
    {"code": "tgl", "name": "Tagalog"},
    {"code": "tha", "name": "Thai"},
    {"code": "tur", "name": "Turkish"},
    {"code": "ukr", "name": "Ukrainian"},
    {"code": "urd", "name": "Urdu"},
    {"code": "uzn", "name": "Northern Uzbek"},
    {"code": "vie", "name": "Vietnamese"}
]

translator = None
model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"

def initialize_translator():
    global translator
    translator = Translator(
        model_name,
        vocoder_name,
        device=torch.device("cuda:0"),
        dtype=torch.float16,
    )

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html', target_lang=target_lang)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        audio_data = request.files['audio_data']
        target_language_code = request.form['target_language_code']

        # Save the audio file in tmp folder
        audio_path = 'tmp/recordedaudio.wav'
        
        # Convert the audio_data to a torchaudio waveform
        waveform, _ = torchaudio.load(io.BytesIO(audio_data.read()), normalize=True)

        # Resample the audio waveform to 16,000 Hz
        resample_transform = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
        resampled_waveform = resample_transform(waveform)

        # Save the resampled waveform to a file
        torchaudio.save(audio_path, resampled_waveform, 16000)

        # Translate audio
        translate_audio(audio_path, target_language_code)

        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, message=str(e))

@app.route('/get_translated_audio')
def get_translated_audio():
    translated_audio_path = "/home/anoni/aitranslate/tmp/translated.wav"
    return send_file(translated_audio_path, as_attachment=True)


def translate_audio(in_file,tgt_lang):
    
    text_output,speech_output = translator.predict(
      input=in_file,
      task_str="s2st",
      tgt_lang=tgt_lang,
    )
    out_file = "/home/anoni/aitranslate/tmp/translated.wav"
    torchaudio.save(out_file, speech_output.audio_wavs[0][0].to(torch.float32).cpu(), speech_output.sample_rate)

if __name__ == '__main__':
    if translator is None:
        initialize_translator()
    app.run(debug=True, use_reloader=False)

