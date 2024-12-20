To train a model to detect a library place .jar or .aar file in data/jars and run fit_gnn.py

To detect libraries semi-dynamically, ensure you have an android device with adb root and frida-server running connected, create a folder 'apks', place target apk in said folder and run detection_pipeline.py. Not 100% this will work in linux, I've Frankensteined
a system with WSL and an emulator running in windows.

To detect libraries statically, rename *.apk to *.zip, copy classes.dex to data/dexs and run dynamic_library_detection.py
