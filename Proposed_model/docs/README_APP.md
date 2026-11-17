# Livestock Counter (PC application)

Counts cattle, goats and sheep in a farm photo. Runs on an ordinary Windows, Linux or Mac laptop. No GPU is needed.

## 1. Files you need in one folder
Unzip `deliverable3_package.zip` (from the Kaggle notebook). Copy these four files from this app zip **into that same folder**:

    app.py   run_app.bat   requirements.txt   README_APP.md

The folder must then contain: `app.py`, `pipeline.py`, `config_d3.json` and `../models/proposed_best.pt`.

## 2. Install (first time only, needs internet)
1. Install Python 3.10 to 3.12 from python.org (tick "Add Python to PATH").
2. Windows: double-click `run_app.bat`. It creates a private environment, installs the libraries and starts the app.
   Other systems, in a terminal inside the folder:

       python -m venv venv
       source venv/bin/activate        (Windows: venv\Scripts\activate)
       pip install -r requirements.txt
       python app.py

The first install downloads about 1 to 2 GB (PyTorch). Later starts are quick.

## 3. Use
1. The browser opens at http://127.0.0.1:7860 (if not, open that address).
2. Upload a farm photo, or drag one into the box.
3. Click **Count animals**.
4. Read the counts and the marked photo. Green boxes are cattle, orange are goats, blue are sheep.
5. Tick **Use tiling** to find small or far-away animals (about 1 second per photo on a laptop CPU). Untick it for the fastest result (about 0.07 s).
6. Raise the confidence slider if there are too many wrong boxes. Lower it if animals are missed.

## 4. How to read the result
The count is an estimate. Photos with crowded herds or many tiny animals may be undercounted, and drawings or toys can be marked by mistake. Check the marked photo before recording the number.

## 5. Notes for the report (fill in your own laptop details)
Operating system, processor, RAM, optional GPU, Python version and library versions: record them from your laptop (Windows: Settings > System > About; `python --version`; `pip list`). Model format: PyTorch weights (`proposed_best.pt`); an ONNX copy (`proposed_best.onnx`) is also in the package.
