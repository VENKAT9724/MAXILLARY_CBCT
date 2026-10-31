# Maxillary Sinus Segmentation - Web App

Local Flask app: upload a CBCT DICOM scan (zip of a series, or a single
.dcm file), runs it through your fine-tuned U-Net, shows overlay results
in the browser, and lets you download all overlays as a zip.

## Setup

1. Install dependencies (Python 3.10+ recommended):
   ```
   pip install -r requirements.txt
   ```
   If you have a CUDA-capable GPU and want to use it, install the matching
   torch build from https://pytorch.org/get-started/locally/ instead of
   the plain `torch` in requirements.txt.

2. Copy your trained checkpoint into the `model/` folder, named exactly:
   ```
   model/best_unet_model_finetuned.pth
   ```
   (Download it from Google Drive first if it's not already on this machine.)

3. Run the app:
   ```
   python app.py
   ```

4. Open `http://localhost:5000` in your browser.

## Notes

- `TARGET_SIZE`, `WL`/`WW` (windowing) in `app.py` must match what was used
  during training/extraction -- currently set to (224, 224) and WL=400/WW=1500
  to match your Colab pipeline. Change these in `app.py` if your setup differs.
- "Process every Nth slice" controls how densely a full volume is sampled;
  a single uploaded .dcm file is always processed fully (it's just 1 slice).
- Uploaded files and intermediate work folders are deleted after each
  request; only the generated overlay PNGs/masks remain, under `results/`.
- This is a local development server (`debug=True`). For anything beyond
  personal/local testing, don't expose this directly to the internet as-is --
  it has no authentication and no production WSGI server (e.g. gunicorn) in front.
