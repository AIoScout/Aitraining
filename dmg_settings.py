import os


application = os.path.join("dist", "TFLiteTraining.app")
files = [application]
symlinks = {"Applications": "/Applications"}

# Volume icon (AIoScouts.icns generated from the repo-root AIoScouts.png).
icon = os.path.join("AIoScouts.icns")
background = None

format = "UDZO"
size = "2G"

window_rect = ((200, 200), (700, 420))
icon_locations = {
    "TFLiteTraining.app": (140, 220),
    "Applications": (520, 220),
}
