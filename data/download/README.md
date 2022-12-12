# How to download the Materials Project CIF files

To replicate our results, place a csv of MP IDS titled 'need.csv' into this directory. You can use any of the dataset files in this repository if renamed.

1. Obtain an API key from the Materials Project database [[link](https://www.materialsproject.org)]
2. Replace the YOUR API KEY placeholder in CIF_grabber.py with your API key
3. Ensure pymatgen is installed (e.g. `pip install pymatgen`)
4. Run the script CIF_grabber.py

This will place all downloaded CIF files in the cifs/ folder. These can then be used by our model. Note that the Materials Project database is contantly updating, so the structures and properties may have changed since the publication of our paper.
