# Prototype AI Sprayer (ESP32-CAM + Laptop TFLite)

## Setup (Windows + PowerShell)

1. Place your TFLite model at `models/mobilenet_v2.tflite` and labels at `models/labels.txt`.
2. Ensure the Python virtual environment exists at `AI_env` (already in repo). If not, create one:

```powershell
python -m venv AI_env
```

3. Install dependencies and run the server:

```powershell
scripts\run_server.ps1 -Port 8000
```

4. Health check:

```powershell
curl http://localhost:8000/health | cat
```

5. Test inference with an image:

```powershell
Invoke-WebRequest -Uri http://localhost:8000/predict -Method POST -InFile .\sample.jpg -ContentType 'multipart/form-data'
```

## ESP32-CAM

- Open `esp32/esp32_cam_relay_post/esp32_cam_relay_post.ino` in Arduino IDE or PlatformIO.
- Set `WIFI_SSID`, `WIFI_PASS`, and `SERVER_URL` (use your laptop IP and port).
- Wire relay to `RELAY_PIN` (default GPIO 12, active-low assumed).

## Notes

- MobileNetV2 input is resized and normalized to [-1, 1].
- Labels should include three classes: `Healthy`, `Mild`, `Severe` in your chosen order.
- Adjust spray threshold/label logic in the ESP32 sketch.
