import win32api

def get_refresh_rate():
    device = win32api.EnumDisplayDevices()
    settings = win32api.EnumDisplaySettings(device.DeviceName, -1)
    return settings.DisplayFrequency

print(f"Monitor refresh rate: {get_refresh_rate()} Hz")
