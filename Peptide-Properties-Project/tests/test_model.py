"""
test_model.py

Basic unit test for the PeptideRegressor model.
"""

import torch
from peptide_properties.model import PeptideRegressor

def test_model_output_shape():
    model = PeptideRegressor(input_dim=20)
    dummy_input = torch.randint(0, 20, (4, 50))  # batch_size=4, sequence_len=50
    output = model(dummy_input)
    assert output.shape == (4, 3), f"Expected output shape (4, 3), but got {output.shape}"
