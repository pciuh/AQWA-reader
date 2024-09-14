# Aqwa Reader

Aqwa Reader is a Python-based tool for reading and processing AQWA files into one file using `pickle`. It includes utility functions for file handling, data extraction, and visualizing results using `matplotlib`. This tool is useful for analyzing data generated by AQWA, a program commonly used in the offshore engineering industry for hydrodynamic analysis.

## Features

- Read and process AQWA output files.
- Extract and visualize data for analysis.
- Supports file compression handling with `zipfile`.
- Customizable and extendable for additional file formats and data handling.

## Requirements

- Python 3.6+
- The following Python libraries:
  - `numpy`
  - `matplotlib`
  - `zipfile`
  
You can install the dependencies using:

```bash
pip install -r requirements.txt
`

## Project Structure:
```bash
aqwa-reader/
│
├── Aqwa_Reader.ipynb       # Main Jupyter notebook with code and examples
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
`
