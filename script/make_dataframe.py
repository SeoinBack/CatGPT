import sys, os
abs_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(abs_path)

import argparse
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.datasets import LmdbDataset, data_list_collater
from fairchem.core.common.relaxation.ase_utils import batch_to_atoms

from ase.io import read
from catgpt.utils.generation_utils import atoms_to_str


def parse_args():
    parser = argparse.ArgumentParser(description='convert data to string')

    parser.add_argument('--name', type=str, help='name of dataset', required=True)
    parser.add_argument('--src_path', type=str, help='path to atomistic data to convert', required=True)
    parser.add_argument('--dst_path', type=str, help='path to string data to save', required=True)
    parser.add_argument('--data_type', type=str, help='lmdb or ase', default='lmdb', choices=['ase','lmdb'])
    
    args = parser.parse_args()

    return args

def convert(args):
    """
    Convert data to dataframe with string representation
    
    to-do: add process to generate invalid catalyst data
    """

    id_list = []
    atoms_list = []
    energy_list = []
    
    # Convert lmdb to ase atoms
    if args.data_type == 'lmdb':
        ads_list = []
        
        with open(abs_path + '/data/mapping/oc20_data_mapping_symbol_only.pkl','rb') as fr:
            mapping = pickle.load(fr)
                    
        dataset = LmdbDataset({'src':args.src_path})
        check_key = True
        key_exists = True
                
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            id_ = data.sid
            energy = data.energy
            
            if check_key:
                if 'random'+str(id_) not in mapping.keys():
                    print(f"Warning: Key 'random{id_}' not found in mapping.")
                    key_exists = False
                    ads = None
                else:
                    ads = mapping['random'+str(id_)]
                    check_key = False
            elif key_exists:
                ads = mapping['random' + str(id_)]
            
            batch = data_list_color([data])
            batch.force = batch.forces
            atoms = batch_to_atoms(batch)
            
            id_list.append(id_)
            atoms_list.append(atoms)
            energy_list.append(energy)
            ads_list.append(ads)
    
    elif args.data_type == 'ase':
        if os.path.isdir(arg.src_path):
            atoms_list = [read(arg.src_path + i) for i in os.listdir(arg.src_path)]
        else:
            atoms_list = read(arg.src_path, ':')
        id_list = list(range(len(atoms_list)))
        
        if 'energy' in atoms_list[0].get_properties([]).keys():
            energy_list = [atoms.get_potential_energy() for atoms in atoms_list]
    
    # Convert atoms to string
    is_tagged =  2 in atoms_list[0].get_tags()
    cat_txt_list = [atoms_to_str(atoms,tagged=is_tagged) for atoms in atoms_list]
    
    df = pd.DataFrame(columns = ['id','cat_txt','target'])
    df['id'] = id_list
    df['cat_txt'] = cat_txt_list
    df['target'] = energy_list
    
    if 'ads_list' in globals():
        df['ads_symbol'] = ads_list
        
    df.to_csv(os.path.join(args.dst_path, args.name + '.csv'))

if __name__ == '__main__':
    args = parse_args()
    convert(args) 

    
    

    
        
        
        
        
        
        
        
            
