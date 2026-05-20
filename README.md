# film-restore-tool
Python scripts to detect and restore film scans in EXR files

https://chatgpt.com/c/69e519ba-a1c0-83e8-9d72-198179ed4d7c

I am building a Python-based GUI tool for restoring scanned film sequences (EXR image sequences). The goal is to detect and repair dust and vertical scratches using temporal information across multiple frames.

Key requirements:

• The application uses PyQt5 for the GUI and OpenImageIO + OpenCV + NumPy for processing.

• The default folder used used to open the file requester uses the path stored on a file at ~.default_dir
• It loads a folder of EXR images and allows stepping through frames interactively.
• Two preview windows are shown side-by-side:

    Left: original frame with optional overlay highlighting detected defects

    Right: restored result
    Core functionality:

    Temporal Restoration

        Use a configurable number of frames (3, 5, or 7)

        The center frame is restored using neighboring frames

        Use temporal median or similar robust method for reconstruction

    Scratch Detection

        Focus on vertical scratches

        Must detect both thin and wide scratches

        Should adapt to scratch width automatically (column-based analysis preferred)

        Detection must be stable across neighboring columns and across frames

    Dust Detection

        Based on temporal differences between frames

        Should avoid false positives from motion
        Performance

    GUI must remain responsive

    Preview should use a fast approximation (single-frame or simplified method)

    Full processing can be slower but should not hang indefinitely

    Frame caching should be used to minimize repeated EXR disk reads

Masking

    Input images include black borders and sprocket holes

    These areas must be excluded from processing automatically

    Prefer automatic detection of valid image area (film gate), not manual painting

GUI Requirements

    Controls on the right side (compact layout)

    Sliders/spinboxes for:

        Dust threshold

        Scratch threshold

        Number of frames (temporal window)

    Overlay toggle for defect visualization

    Synchronized pan/zoom between both preview windows

    Progress bar and status feedback during processing

    Cancel button for long operations

Processing Output

    Writes restored EXR sequence to a subfolder

    Must provide clear console/log feedback during processing


Basic structure:

restore_tool/

├── main.py              # entry point

├── app.py               # GUI + controller

├── viewer.py            # image viewer widget


├── io_utils.py          # EXR read/write + default_dir

├── cache.py             # frame caching



├── processing/


│   ├── restore.py       # restore_frame()

│   ├── scratch.py       # scratch detection

│   ├── dust.py          # dust detection

│   ├── mask.py          # auto film gate detection

│   └── preview.py       # fast preview version


Responsibilities by Module
io_utils.py
    handles:

    read_exr(path)

    write_exr(path, img)

    load_default_dir()

    save_default_dir()

cache.py

    Handles:

    LRU frame cache

    prefetch neighbors

    Interface:

    frame = cache.get(i)

mask.py

    def auto_detect_mask(img):
        return mask

    runs ONCE at load

    reused everywhere

scratch.py

    def detect_scratches(gray, params):
        return mask

    Contains:

    multiscale detection

    width adaptation

    column smoothing

    (later) temporal stabilization

dust.py

    def detect_dust(frames, center, thresh):
        return mask


restore.py

    def restore_frame(frames, p):
        return out, mask
    core pipeline, clean and readable:

    scratch = temporal_scratch(...)
    dust    = detect_dust(...)
    mask    = combine

    temporal = median(frames)
    repair   = inpaint

    out = blend

preview.py

    def restore_preview(frame, p):
       return out, mask

    NO temporal median

    NO heavy loops

    must be fast (<50ms)

viewer.py

    ONLY:

    display image

    pan/zoom sync

    overlay drawing
    
    No processing logic


app.py (Controller)

    This is the brain:

    Handles:

    UI events

    parameter updates

    calling processing

    applying masks

    managing cache

