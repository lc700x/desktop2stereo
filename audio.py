import pyaudio
import wave

def record_audio_to_file(filename="output.wav", seconds=5, device_index=1):
    p = pyaudio.PyAudio()

    # List devices
    print("Available audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"{i}: {info['name']}")
    exit()
    # Stream config
    format = pyaudio.paInt16
    channels = 2
    rate = 44100
    chunk = 1024

    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk,
                    input_device_index=device_index)

    print(f"Recording for {seconds} seconds...")
    frames = []

    for _ in range(0, int(rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("Done recording.")

    # Stop and close
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save as WAV
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Saved recording to {filename}")

# Example usage:
record_audio_to_file("test_output.wav", seconds=5, device_index=1)
