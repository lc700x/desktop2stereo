import sounddevice as sd

# Get a list of all available devices
devices = sd.query_devices()

# Print information about each device
for i, device in enumerate(devices):
    print(f"Device {i}: {device['name']}")
    print(f"  Host API: {device['hostapi']}")
    print(f"  Input Channels: {device['max_input_channels']}")
    print(f"  Output Channels: {device['max_output_channels']}")
    print(f"  Default Sample Rate: {device['default_samplerate']}\n")