# Plot rocket launch data from rocket launch videos

## General
![Acceleration plot](examples/ift5_acc.png?raw=true)

Using PyTesseract for extracting the telemetry data from SpaceX, Rocket Lab and Arianespace launch videos.
Velocity, altitude and acceleration are then plotted for each stage (SpaceX) or for the main stage (Rocket Lab, Arianespace).
Outliers are detected and ignored by applying acceleration and vertical speed boundaries.
Realtime performance can be reached by only analysing every nth frame. Accelerations are the combination of velocity change
rates and acceleration due to gravitational forces. Furthermore, accelerations are shown as a moving average.

## Arguments
Arguments: `--video` (Video local path), `--start` (Start time in video in seconds), `--duration` (Duration of video from start time). The 
supported time formats are: 1:13:12, 3:12, 144 (h:min:s, min:s, s). The `--type` flag specifies where the text detector looks for velocity and altitude data, while the plot title can be set with `--title`.

Example: `python rocket_data.py --video ift5-1080p.mp4 --start 0:24 --duration 9:52 --type SpaceX --title IFT-5`

## Installation
Tesseract must be installed on the system and referenced, installation link for Windows: https://github.com/UB-Mannheim/tesseract/wiki/Windows-build

On Ubuntu install with:
```
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

## Examples
### Starship IFT-5
![Velocity plot](examples/ift5_velo.png?raw=true)
![Altitude plot](examples/ift5_alti.png?raw=true)
![Acceleration plot](examples/ift5_acc.png?raw=true)
### Inspiration4
![Velocity plot](examples/inspiration4_velo.png?raw=true)
![Altitude plot](examples/inspiration4_alti.png?raw=true)
![Acceleration plot](examples/inspiration4_acc.png?raw=true)

### Double Asteroid Redirection Test (DART)
![Velocity plot](examples/dart_velo.png?raw=true)
![Altitude plot](examples/dart_alti.png?raw=true)
![Acceleration plot](examples/dart_acc.png?raw=true)

### Transporter-3
![Velocity plot](examples/transporter3_velo.png?raw=true)
![Altitude plot](examples/transporter3_alti.png?raw=true)
![Acceleration plot](examples/transporter3_acc.png?raw=true)
