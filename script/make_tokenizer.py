import numpy as np
import argparse
import os

import tokenizers
from transformers import PreTrainedTokenizerFast
    
def parse_args():
    parser = argparse.ArgumentParser(description='generate tokenizer for CatGPT')

    parser.add_argument('--data_path', type=str, help='base directory to save tokenizer files', required=True)
    parser.add_argument(
        '--token_type', 
        type=str, 
        help='type of method to tokenize catalysts', 
        required=True, 
        choices=['digit','coordinate','ads']
    )
    parser.add_argument('--max_length', type=int, help='max token length', default=1024)
    
    args = parser.parse_args()

    return args
    
def make_tokenizer(args):
    
    max_length = args.max_length
    token_type = args.token_type
    data_path = os.path.join(args.data_path, f'{token_type}-tokenizer/')
    os.makedirs(data_path, exist_ok=True)
    
    # Elemental symbols
    ATOMS = ["Si", "C", "Pb", "I", "Br", "Cl", "Eu", "O", "Fe", "Sb", "In", "S", "N", "U", "Mn", "Lu", "Se", "Tl", "Hf",
             "Ir", "Ca", "Ta", "Cr", "K", "Pm", "Mg", "Zn", "Cu", "Sn", "Ti", "B", "W", "P", "H", "Pd", "As", "Co", "Np",
             "Tc", "Hg", "Pu", "Al", "Tm", "Tb", "Ho", "Nb", "Ge", "Zr", "Cd", "V", "Sr", "Ni", "Rh", "Th", "Na", "Ru",
             "La", "Re", "Y", "Er", "Ce", "Pt", "Ga", "Li", "Cs", "F", "Ba", "Te", "Mo", "Gd", "Pr", "Bi", "Sc", "Ag", "Rb",
             "Dy", "Yb", "Nd", "Au", "Os", "Pa", "Sm", "Be", "Ac", "Xe", "Kr", "He", "Ne", "Ar"]

    # Digits for lattices and fractiaonl coordinates
    DIGITS = [str(d) for d in list(range(10))] # '0' ~ '9'
    COORDINATES = ["{0:.3f}".format(d) for d in list(np.linspace(0,1,1001))]
    
    
    # Symbols for decimal and etc
    SYMBOLS = ['.']
    
    ADS_SYMBOLS = ['*OHCH2CH3', '*ONN(CH3)2', '*H', '*CH*CH', '*COHCOH', '*OCHCH3', '*NO2NO2', 'CH2*CO', '*CH4', '*NHNH',
                   '*CCH', '*CH2CH2OH', '*CHOHCH3', '*CHCH2OH', '*ONNH2', '*C*C', '*CCH2OH', '*CCO', '*CN', '*CHOCHOH',
                   '*NO3', '*OHNH2', '*N*NH', '*CH2CH3', '*COHCH3', '*C', '*NH3', '*CCH2', '*NH', '*CHCHO', '*CH*COH', 
                   '*CHOH', '*CHOHCHOH', '*COHCHOH', '*CHCH2', '*CHOCH2OH', '*CCHO', '*N', '*N*NO', '*OH', '*COCH2O', 
                   '*COCH3', '*CCH3', '*OHCH3', '*CH2', '*CH3', '*CH2*O', '*OCH2CH3', '*CH2OH', '*CCHOH', '*OCH3', 
                   '*CHO*CHO', '*CHOHCH2OH', '*CHCHOH', '*OCH2CHOH', '*OH2', '*OHNNCH3', '*N2', '*COHCH2OH', '*COCHO', 
                   '*NO2', '*O', '*CHCO', '*COHCHO', '*CHOHCH2', '*ONH', '*NONH', '*NO']
    
    special_tokens = ["<bos>", "<eos>", "<pad>", "<unk>", "<mask>"]

    if token_type == 'digit':
        VOCAB = SYMBOLS + DIGITS + ATOMS
    
    elif token_type == 'coordinate':
        VOCAB = SYMBOLS + COORDINATES + ATOMS
    
    elif token_type == 'ads':
        VOCAB = SYMBOLS + COORDINATES + ATOMS + ADS_SYMBOLS
    
    else:
        raise TypeError(f'Invalid token_type "{token_type}"')
    
    # Construct vocabulary dictionary
    VOCAB_dict = {v: i for i, v in enumerate(special_tokens + VOCAB)}
    
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=VOCAB_dict, unk_token='<unk>'))

    # Configure pre-tokenizers
    if token_type == 'digit':
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
            tokenizers.pre_tokenizers.Whitespace(),
            tokenizers.pre_tokenizers.Digits(individual_digits=True)
        ])
    
    else:
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
            tokenizers.pre_tokenizers.WhitespaceSplit(),
        ])
    
    tokenizer.add_special_tokens(special_tokens)

    # Enable truncation
    tokenizer.enable_truncation(max_length=max_length)
    
    # Create a PreTrainedTokenizerFast object from the tokenizer
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token='<bos>',
        eos_token='<eos>',
        unk_token='<unk>',
        pad_token='<pad>',
        mask_token='<mask>',
        vocab_size=len(tokenizer.get_vocab()),
        max_len = max_length
    )    
            
    wrapped_tokenizer.save_pretrained(data_path)
    print(f"{token_type}-tokenizer saved at: {data_path}")
    
if __name__ == '__main__':
    args = parse_args()
    make_tokenizer(args) 
