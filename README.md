# Peptide-Properties

A deep learning pipeline to predict molecular properties of peptides using their amino acid sequences.

## Author
Varun Chandrasekar

## Project Description
This project reads peptide sequences from a FASTA file, computes target properties (molecular weight, pI, instability index), and uses a PyTorch model to predict these values.

## Folder Structure
```
Peptide-Properties-Project/
├── peptide_properties/
│   ├── model.py
│   └── preprocess.py
├── notebooks/
│   └── Peptide-Properties.ipynb
├── main.py
├── requirements.txt
├── tests/
│   └── test_model.py
├── README.md
```

## How to Use

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
python main.py
```

## Running Tests

To run basic unit tests for the model, install `pytest` and execute:

```bash
pip install pytest
pytest tests/
```

## Notes
The required peptide FASTA file will be automatically downloaded from a public repository if not already present.
