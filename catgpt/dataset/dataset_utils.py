import pickle
import random
from math import gcd
from functools import partial, reduce

def load_pickle(src):
    with open(src,'rb') as fr:
        return pickle.load(fr)

def to_split(input_str):
    input_str_list = input_str.split(' ')
    return ' '.join(['_' + i for i in input_str_list[:6]] + input_str_list[6:])
    
def to_coordinate(input_str):
    return input_str
    
def to_ads(input_str, ads):
    return ' '.join([ads,input_str])
    
def to_digit(input_str):
    input_str_list = input_str.split(' ')
    digit_str =  ''
    for idx, comp in enumerate(''.join(input_str_list[:6])):
        if idx == 0:
            digit_str += comp
        else:
            digit_str += ' ' + comp
    
    for idx, comp in enumerate(input_str_list[6:]):
        if idx%4 == 0:
            digit_str += ' ' + comp
        else:
            digit_str += ' ' + ' '.join([i for i in comp])
            
    return digit_str
        

def permutation(input_str, ads):
    input_str_list = input_str.split(' ')
    num_ads = count_ads_atoms(ads)
    
    lattice = input_str_list[:6]
    slab_atoms = input_str_list[6:-4*num_ads]
    ads_atoms = input_str_list[-4*num_ads:]
    
    permuted_slab_atoms = []
    for i in range(int(len(slab_atoms)/4)):
        permuted_slab_atoms.append(' '.join(slab_atoms[i*4:i*4+4]))
        
    random.shuffle(permuted_slab_atoms)
    
    return ' '.join(lattice + permuted_slab_atoms + ads_atoms)
    

def str_preprocess(string_type, input_str, augment_type=None, **kwargs):
    #if augment_type == 'permutation':
    #    ads = kwargs.get('ads', None)
    #    input_str = permutation(input_str, ads)

    if string_type in ['coordinate','t5']:
        return to_coordinate(input_str)
        
    elif string_type == 'split':
        return to_split(input_str)
    
    #elif string_type == 'ads':
    #    ads = kwargs.get('ads', None)
    #    return to_ads(input_str, ads)
    
    elif string_type == 'digit':
        return to_digit(input_str)

def prop_preprocess(input_dict):
    props = ['ads','comp','spg','miller','energy']
    prop_str_list = []
    for prop in props:
        if prop in input_dict.keys():
            prop_str_list.append(input_dict[prop])
            
    prop_str = ' '.join(prop_str_list)
    return prop_str

def count_ads_atoms(adsorbate_symbol: str) -> int:
    def parse_formula(formula: str) -> dict:
        stack = [{}]
        i = 0
        n = len(formula)
        
        while i < n:
            if formula[i] == '(':
                stack.append({})
                i += 1
            elif formula[i] == ')':
                top = stack.pop()
                i += 1
                start = i
                while i < n and formula[i].isdigit():
                    i += 1
                multiplier = int(formula[start:i] or '1')
                for elem, cnt in top.items():
                    stack[-1][elem] = stack[-1].get(elem, 0) + cnt * multiplier
            else:
                elem = formula[i]
                i += 1
                start = i
                while i < n and formula[i].isdigit():
                    i += 1
                count = int(formula[start:i] or '1')
                stack[-1][elem] = stack[-1].get(elem, 0) + count
        
        return stack[0]
    
    clean_formula = adsorbate_symbol.replace('*', '')
    atom_counts = parse_formula(clean_formula)
    total_atoms = sum(atom_counts.values())
    
    return total_atoms
    
    
def simplify_formula(s):
    tokens = s.split()
    numbers = []
    for i, token in enumerate(tokens):
        if i % 2 == 0:
            try:
                int(token)
                return s
            except ValueError:
                continue
        else:

            try:
                n = int(token)
                numbers.append(n)
            except ValueError:
                return s

    if len(numbers) < 2:
        return s

    overall_gcd = reduce(gcd, numbers)
    if overall_gcd == 1:
        return s

    new_tokens = []
    num_idx = 0
    for i, token in enumerate(tokens):
        if i % 2 == 0:
            new_tokens.append(token)
        else:
            new_tokens.append(str(numbers[num_idx] // overall_gcd))
            num_idx += 1
    return " ".join(new_tokens)