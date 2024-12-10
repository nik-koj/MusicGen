from flask import Flask, render_template, request, url_for, send_file
from transformers import AutoProcessor, MusicgenForConditionalGeneration, pipeline
import scipy.io.wavfile
from googletrans import Translator
import numpy as np
import os

app = Flask(__name__)

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

translator = Translator()

if not os.path.exists("static/generated"):
    os.makedirs("static/generated")


@app.route("/", methods=["GET", "POST"])
def index():
    music_file_url = None
    if request.method == "POST":
        try:
            user_text = request.form["user_text"]
            user_duration = int(request.form["user_duration"])
            genre = request.form["genre"]
            mood = request.form["mood"]

            translated_genre = translator.translate(genre, src='ru', dest='en').text
            translated_mood = translator.translate(mood, src='ru', dest='en').text
            translated_text = translator.translate(user_text, src='ru', dest='en').text

            music_prompt = f"{translated_genre}, {translated_mood}, {translated_text}"
            tokens_per_second = 50
            max_length = user_duration * tokens_per_second

            music_inputs = processor(text=[music_prompt], padding=True, return_tensors="pt")
            audio_tokens = music_model.generate(**music_inputs, max_length=max_length)
            audio_waveform = audio_tokens[0, 0].cpu().numpy()
            audio_waveform = (audio_waveform * 32767).astype("int16")

            sampling_rate = music_model.config.audio_encoder.sampling_rate
            file_name = f"musicgen_out_{len(os.listdir('static/generated')) + 1}.wav"
            wav_file_path = os.path.join("static/generated", file_name)
            scipy.io.wavfile.write(wav_file_path, rate=sampling_rate, data=audio_waveform)

            music_file_url = url_for("static", filename=f"generated/{file_name}")
        except Exception as e:
            return f"Обнаружена ошибка: {e}"

    return render_template("index.html", music_file_url=music_file_url)


@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join("static/generated", filename)
    return send_file(file_path, mimetype="audio/wav", download_name=filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
