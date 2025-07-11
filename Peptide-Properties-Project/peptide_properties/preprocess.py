"""
preprocess.py

Parses FASTA sequences and computes peptide-level features.
"""

import re
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_int = {aa: idx for idx, aa in enumerate(amino_acids)}

def is_standard_sequence(seq):
    pattern = f'[{amino_acids}]+'
    return re.fullmatch(pattern, seq) is not None

def load_sequences(fasta_file):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq).replace("-", "")
        if is_standard_sequence(seq):
            sequences.append(seq)
    return sequences

def compute_targets(sequences):
    targets = []
    for seq in sequences:
        analysis = ProteinAnalysis(seq)
        mw = analysis.molecular_weight()
        pI = analysis.isoelectric_point()
        instability = analysis.instability_index()
        aromaticity = analysis.aromaticity()
        targets.append([mw, pI, instability, aromaticity])
    return targets
