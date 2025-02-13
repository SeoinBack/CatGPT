import numpy as np
import argparse
import os

import tokenizers

from tokenizers import PreTokenizedString
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
    
def parse_args():
    parser = argparse.ArgumentParser(description='generate tokenizer for CatGPT')

    parser.add_argument('--data-path', type=str, help='base directory to save tokenizer files', required=True)
    parser.add_argument(
        '--token-type', 
        type=str, 
        help='type of method to tokenize catalysts', 
        required=True, 
        choices=['digit','coordinate','t5']
    )
    parser.add_argument('--add-props', action='store_true', help='weather to add property tokens', default=False)
    parser.add_argument('--max-length', type=int, help='max token length', default=1024)
    
    args = parser.parse_args()

    return args
    
def make_tokenizer(args):
    
    if args.add_props:
        prop = 'prop-'
    else:
        prop = ''
    
    max_length = args.max_length
    token_type = args.token_type
    data_path = os.path.join(args.data_path, f'{token_type}-{prop}tokenizer/')
    os.makedirs(data_path, exist_ok=True)
    
    # Elemental symbols
    ATOMS = ["Si", "C", "Pb", "I", "Br", "Cl", "Eu", "O", "Fe", "Sb", "In", "S", "N", "U", "Mn", "Lu", "Se", "Tl", "Hf",
             "Ir", "Ca", "Ta", "Cr", "K", "Pm", "Mg", "Zn", "Cu", "Sn", "Ti", "B", "W", "P", "H", "Pd", "As", "Co", "Np",
             "Tc", "Hg", "Pu", "Al", "Tm", "Tb", "Ho", "Nb", "Ge", "Zr", "Cd", "V", "Sr", "Ni", "Rh", "Th", "Na", "Ru",
             "La", "Re", "Y", "Er", "Ce", "Pt", "Ga", "Li", "Cs", "F", "Ba", "Te", "Mo", "Gd", "Pr", "Bi", "Sc", "Ag", "Rb",
             "Dy", "Yb", "Nd", "Au", "Os", "Pa", "Sm", "Be", "Ac", "Xe", "Kr", "He", "Ne", "Ar"]

    # Digits for lattices and fractiaonl coordinates
    DIGITS = [str(d) for d in list(range(10))] # '0' ~ '9'
    COORDINATES = ['{0:.3f}'.format(d) for d in list(np.linspace(0,1,1001))]
    ENERGIES = ['{0:.1f}'.format(d) for d in np.round(np.arange(-10.0,10.1,0.1),1)]
    
    # Symbols for decimal and etc
    SYMBOLS = ['.']
    
    ADS_SYMBOLS = ['*OHCH2CH3', '*ONN(CH3)2', '*H', '*CH*CH', '*COHCOH', '*OCHCH3', '*NO2NO2', 'CH2*CO', '*CH4', '*NHNH',
                   '*CCH', '*CH2CH2OH', '*CHOHCH3', '*CHCH2OH', '*ONNH2', '*C*C', '*CCH2OH', '*CCO', '*CN', '*CHOCHOH',
                   '*NO3', '*OHNH2', '*N*NH', '*CH2CH3', '*COHCH3', '*C', '*NH3', '*CCH2', '*NH', '*CHCHO', '*CH*COH', 
                   '*CHOH', '*CHOHCHOH', '*COHCHOH', '*CHCH2', '*CHOCH2OH', '*CCHO', '*N', '*N*NO', '*OH', '*COCH2O', 
                   '*COCH3', '*CCH3', '*OHCH3', '*CH2', '*CH3', '*CH2*O', '*OCH2CH3', '*CH2OH', '*CCHOH', '*OCH3', 
                   '*CHO*CHO', '*CHOHCH2OH', '*CHCHOH', '*OCH2CHOH', '*OH2', '*OHNNCH3', '*N2', '*COHCH2OH', '*COCHO', 
                   '*NO2', '*O', '*CHCO', '*COHCHO', '*CHOHCH2', '*ONH', '*NONH', '*NO']
    
    SPACE_GROUPS = ['P1', 'P-1', 'P121', 'P12_11', 'C121', 'P1m1', 'P1c1', 'C1m1', 'C1c1', 'P12/m1', 'P12_1/m1', 
                   'C12/m1', 'P12/c1', 'P12_1/c1', 'C12/c1', 'P222', 'P222_1', 'P2_12_12', 'P2_12_12_1', 'C222_1', 
                   'C222', 'F222', 'I222', 'I2_12_12_1', 'Pmm2', 'Pmc2_1', 'Pcc2', 'Pma2', 'Pca2_1', 'Pnc2', 'Pmn2_1', 
                   'Pba2', 'Pna2_1', 'Pnn2', 'Cmm2', 'Cmc2_1', 'Ccc2', 'Amm2', 'Aem2', 'Ama2', 'Aea2', 'Fmm2', 'Fdd2', 
                   'Imm2', 'Iba2', 'Ima2', 'Pmmm', 'Pnnn', 'Pccm', 'Pban', 'Pmma', 'Pnna', 'Pmna', 'Pcca', 'Pbam', 'Pccn',
                   'Pbcm', 'Pnnm', 'Pmmn', 'Pbcn', 'Pbca', 'Pnma', 'Cmcm', 'Cmce', 'Cmmm', 'Cccm', 'Cmme', 'Ccce', 'Fmmm',
                   'Fddd', 'Immm', 'Ibam', 'Ibca', 'Imma', 'P4', 'P4_1', 'P4_2', 'P4_3', 'I4', 'I4_1', 'P-4', 'I-4', 'P4/m',
                   'P4_2/m', 'P4/n', 'P4_2/n', 'I4/m', 'I4_1/a', 'P422', 'P42_12', 'P4_122', 'P4_12_12', 'P4_222', 'P4_22_12', 
                   'P4_322', 'P4_32_12', 'I422', 'I4_122', 'P4mm', 'P4bm', 'P4_2cm', 'P4_2nm', 'P4cc', 'P4nc', 'P4_2mc',
                   'P4_2bc', 'I4mm', 'I4cm', 'I4_1md', 'I4_1cd', 'P-42m', 'P-42c', 'P-42_1m', 'P-42_1c', 'P-4m2', 'P-4c2',
                   'P-4b2', 'P-4n2', 'I-4m2', 'I-4c2', 'I-42m', 'I-42d', 'P4/mmm', 'P4/mcc', 'P4/nbm', 'P4/nnc', 'P4/mbm', 
                   'P4/mnc', 'P4/nmm', 'P4/ncc', 'P4_2/mmc', 'P4_2/mcm', 'P4_2/nbc', 'P4_2/nnm', 'P4_2/mbc', 'P4_2/mnm', 
                   'P4_2/nmc', 'P4_2/ncm', 'I4/mmm', 'I4/mcm', 'I4_1/amd', 'I4_1/acd', 'P3', 'P3_1', 'P3_2', 'R3', 'P-3', 
                   'R-3', 'P312', 'P321', 'P3_112', 'P3_121', 'P3_212', 'P3_221', 'R32', 'P3m1', 'P31m', 'P3c1', 'P31c', 
                   'R3m', 'R3c', 'P-31m', 'P-31c', 'P-3m1', 'P-3c1', 'R-3m', 'R-3c', 'P6', 'P6_1', 'P6_5', 'P6_2', 'P6_4',
                   'P6_3', 'P-6', 'P6/m', 'P6_3/m', 'P622', 'P6_122', 'P6_522', 'P6_222', 'P6_422', 'P6_322', 'P6mm', 
                   'P6cc', 'P6_3cm', 'P6_3mc', 'P-6m2', 'P-6c2', 'P-62m', 'P-62c', 'P6/mmm', 'P6/mcc', 'P6_3/mcm', 
                   'P6_3/mmc', 'P23', 'F23', 'I23', 'P2_13', 'I2_13', 'Pm-3', 'Pn-3', 'Fm-3', 'Fd-3', 'Im-3', 'Pa-3', 
                   'Ia-3', 'P432', 'P4_232', 'F432', 'F4_132', 'I432', 'P4_332', 'P4_132', 'I4_132', 'P-43m', 'F-43m', 
                   'I-43m', 'P-43n', 'F-43c', 'I-43d', 'Pm-3m', 'Pn-3n', 'Pm-3n', 'Pn-3m', 'Fm-3m', 'Fm-3c', 'Fd-3m', 
                   'Fd-3c', 'Im-3m', 'Ia-3d']
    
    MILLER_INDICES = ['(1,1,2)', '(2,-2,-1)', '(2,2,-1)', '(1,-1,0)', '(2,1,0)', '(1,1,1)', '(2,0,-1)', '(1,0,0)', '(1,0,1)', 
                      '(1,0,2)', '(1,-1,2)', '(1,-1,1)', '(1,-1,-2)', '(1,1,-2)', '(2,2,1)', '(1,1,-1)', '(2,1,-1)', 
                      '(2,-1,0)', '(1,2,-2)', '(1,1,0)', '(0,0,1)', '(1,2,1)', '(1,2,2)', '(2,-2,1)', '(0,1,1)', 
                      '(1,-2,-1)', '(0,1,0)', '(0,2,-1)', '(2,-1,1)', '(1,2,-1)', '(1,0,-2)', '(1,-2,2)', '(0,1,-2)', 
                      '(1,-2,-2)', '(1,2,0)', '(2,-1,-1)', '(2,1,1)', '(0,2,1)', '(1,-2,1)', '(1,0,-1)', '(2,-1,2)', 
                      '(0,1,-1)', '(2,0,1)', '(0,1,2)', '(1,-2,0)', '(2,-1,-2)', '(1,-1,-1)', '(2,1,2)', '(2,1,-2)']
    
    EXTRA_IDS = [f"<extra_id_{i}>" for i in range(20)]
    
    special_tokens = ["<bos>", "<eos>", "<pad>", "<unk>", "<mask>"]

    if token_type == 'digit':
        VOCAB = SYMBOLS + DIGITS + ATOMS
    
    elif token_type == 'coordinate':
        VOCAB = SYMBOLS + COORDINATES + ATOMS
    
    elif token_type == 't5':
        VOCAB = SYMBOLS + COORDINATES + ATOMS
        special_tokens = special_tokens + EXTRA_IDS
    
    else:
        raise TypeError(f'Invalid token_type "{token_type}"')
    
    if args.add_props:
        VOCAB += ADS_SYMBOLS + SPACE_GROUPS + MILLER_INDICES
        if token_type != 'digit':
            VOCAB += DIGITS + ENERGIES
    
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
    print(f"{token_type}-{prop}tokenizer saved at: {data_path}")
  
if __name__ == '__main__':
    args = parse_args()
    make_tokenizer(args) 
