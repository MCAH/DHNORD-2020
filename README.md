# DHNORD2020

This repository is presented as a part of the Media Center for Art History's presentation at the [#dhnord2020](https://www.meshs.fr/page/dhnord2020) colloquium, *[The Art Historical Image Collection at Columbia University: Automating Research on its Construction and Creators](https://www.meshs.fr/page/the_art_historical_image_collection_at_columbia_university)*.

## Usage

### Input

Create a folder for images prepared for classification. If using scanned slide images, scans should be cropped to isolate transparency. Images can be stored in a different directory than the script however only images should be included in working folders. For a large quantity of images, it is recommended to run the script in batches.

### Required Packages

```
pip install numpy
pip install pandas
pip install xgboost==0.90
pip install tk
pip install opencv-python
pip install joblib
pip install sklearn
pip install matplotlib
pip install tqdm
pip install scipy
```

### Running  Instructions

Ensure `finalized_model_xgboost_08_19.joblib.dat` and `xgb_pred_Visualizations_mcah_error_catch.py` are in the same folder.

Run the python file:
```
python3 xgb_pred_Visualizations_mcah_error_catch.py
```

When prompted, select the folder containing images prepared earlier.

### Output

The script will produce:

* `[image_folder_name]_DFT_0`
    * This folder contains DFT images used for classification.
* `results_[image_folder_name].csv`
    * This CSV presents the halftone or non-halftone classification for each image. A probability is included.
* `[image_folder_name]_Processed_Visualizations_0`
    * This folder contains images of visualizations of the 'sparkles' on each DFT and a CSV with location data for each point. 

## Contact

mediacenter@columbia.edu

## License

Copyright 2021 The Trustees of Columbia University, Media Center for Art History, Department of Art History & Archaeology.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.