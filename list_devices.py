import sounddevice as sd

print("Available audio devices:")
devices = sd.query_devices()
for i, device in enumerate(devices):
    print(
        f"{i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})"
    )

print("\nDefault input device:", sd.default.device[0])
print("Default output device:", sd.default.device[1])
