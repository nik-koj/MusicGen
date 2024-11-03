from flask import Flask, render_template, request, url_for, send_file
from transformers import AutoProcessor, MusicgenForConditionalGeneration, pipeline
import scipy.io.wavfile
import numpy as np
import os

app = Flask(__name__)

# Load the music generation model and processor
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Load the lyrics generation pipeline
lyrics_pipe = pipeline("text-generation", model="ECE1786-AG/lyrics-generator")

# Save generated files to a folder
if not os.path.exists("static/generated"):
    os.makedirs("static/generated")

@app.route("/", methods=["GET", "POST"])
def index():
    music_file_url = None
    generated_lyrics = None  # Initialize lyrics variable
    if request.method == "POST":
        user_text = request.form["user_text"]
        user_duration = int(request.form["user_duration"])
        genre = request.form["genre"]
        mood = request.form["mood"]

        if user_text:
            # Create a prompt for generating lyrics
            prompt = f"Generate lyrics of a song based on these words: {genre}, {mood}, {user_text}"
            
            # Generate lyrics using the lyrics generation model
            generated_lyrics = lyrics_pipe(prompt, max_length=500, num_return_sequences=1)[0]['generated_text']
            generated_lyrics = generated_lyrics[len(user_text):]

            # Generate music based on a different prompt
            music_prompt = f"{genre}, {mood}, {user_text}"
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

    return render_template("index.html", music_file_url=music_file_url, generated_lyrics=generated_lyrics)

@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join("static/generated", filename)
    return send_file(file_path, mimetype="audio/wav", download_name=filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
