# Pneumonia Detection Using CNN (Streamlit)

Educational demo. **Not a medical diagnostic tool.** Consult a healthcare professional for any medical concern.

## Structure
```
Pneumonia-Detection/
├── app.py
├── trained.h5                          # your trained model (from the repo)
├── Pneumonia_detection_using_CNN.ipynb # your original notebook (unchanged)
├── requirements.txt
└── README.md
```

## Install
```bash
git clone https://github.com/Aryan-Kr-Ve/Pneumonia-detection-using-Convolutional-neural-networks.git
cd Pneumonia-detection-using-Convolutional-neural-networks
# copy app.py and requirements.txt into this folder
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Verify these 2 settings (top of app.py)
Input size, colour mode and output type are read from the model automatically. Two things cannot be, so confirm them in your notebook:

1. `PIXEL_SCALE` - did training use `rescale=1./255` (keep 255.0) or no scaling (set 1.0)?
2. `CLASS_NAMES` order - run this in the notebook after training/loading:
   ```python
   print(train_generator.class_indices)   # e.g. {'NORMAL': 0, 'PNEUMONIA': 1}
   ```
   Keep `["Normal", "Pneumonia"]` if NORMAL is 0; otherwise swap them.

If you didn't use generators, check how labels were built (which class got 0 and 1) and how images were resized/normalised.

## Troubleshooting
- **Model file not found:** put `trained.h5` next to `app.py` (or in `model/`) and run from that folder.
- **Model fails to load / deserialization error:** the `.h5` was saved with a different TensorFlow/Keras version. Install the version used in your notebook (`pip install tensorflow==<version>`), or re-save the model in the notebook with `model.save("trained.keras")` and update `MODEL_CANDIDATES`.
- **`No module named tensorflow`:** TensorFlow needs a supported Python version (3.9-3.12); create a venv with one.
- **Wrong or always-the-same predictions:** re-check the two settings above.
- **Large `.h5` not on disk after clone:** if it is stored with Git LFS, run `git lfs pull`.
