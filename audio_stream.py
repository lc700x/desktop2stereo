from flask import Flask, Response
import pyaudio
import wave
import io

app = Flask(__name__)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
DEVICE_INDEX = 1

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=DEVICE_INDEX)

def generate_wav_stream():
    """Generate a WAV stream continuously from microphone."""
    wav_buffer = io.BytesIO()
    wf = wave.open(wav_buffer, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b'')  # Initialize WAV header
    wf.close()
    wav_header = wav_buffer.getvalue()[:44]  # standard WAV header size

    # send header first
    yield wav_header

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        yield data

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>Live Audio</title></head>
    <body>
        <h1>Live Audio Stream</h1>
        <audio controls autoplay>
            <source src="/audio_stream" type="audio/wav">
        </audio>
    </body>
    </html>
    '''

@app.route('/audio_stream')
def audio_stream():
    return Response(generate_wav_stream(), mimetype="audio/wav")

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
