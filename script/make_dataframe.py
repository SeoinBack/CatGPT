import sys, os
abs_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(abs_path)

import argparse
import pandas as pd
import numpy as np
import pickle
import random

from tqdm import tqdm

from fairchem.core.datasets import LmdbDataset, data_list_collater
from fairchem.core.common.relaxation.ase_utils import batch_to_atoms

from ase.io import read
from catgpt.utils.generation_utils import atoms_to_str, str_to_atoms
from sklearn.model_selection import train_test_split

random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='convert data to string')

    parser.add_argument('--name', type=str, help='name of dataset', required=True)
    parser.add_argument('--src-path', type=str, help='path to atomistic data to convert', required=True)
    parser.add_argument('--dst-path', type=str, help='path to string data to save', required=True)
    parser.add_argument('--data-type', type=str, help='lmdb or ase', default='lmdb', choices=['ase','lmdb'])
    parser.add_argument('--corrupt-data', 
                        action='store_true', 
                        help='whether to add corrupted data for training detectino model', 
                        default=False
                        )  
    parser.add_argument('--props',
                        nargs='+',
                        choices=['spg','miller','energy','comp'],
                        help="list of properties to include dataframe, separated by spaces (e.g., --props spg mil energy)"
                        )
    args = parser.parse_args()
    return args

def corrupt_atoms(atoms):

    n_atoms = len(atoms)
    copy_atoms = atoms.copy()
    rand = random.random()
    
    if rand > 0.5:
        # delete
        n_rand = random.choice(range(int(n_atoms*0.2),int(n_atoms*0.8)))
        rand_idx = random.sample(range(len(atoms)),n_rand)
        del copy_atoms[rand_idx]
        
        return copy_atoms, 'delete', rand_idx
    else:
        #scaling
        scale_rand = random.random()/2 + 1.5
        copy_atoms.set_cell(copy_atoms.get_cell()*scale_rand,scale_atoms=True)
        
        return copy_atoms, 'scaling', scale_rand

def corrupt_data(cat_txt_list):
    clean_idx, corrupt_idx = train_test_split(
        range(len(cat_txt_list)), 
        test_size=0.5, 
        random_state=42
    )
    
    corrupted_cat_txt_list = []
    corruption_label_list = []
    corruption_type_list = []
    parameter_list = []
    
    for idx in tqdm(range(len(cat_txt_list)), desc='Corrupting'):
        cat_txt = cat_txt_list[idx]
        if idx in clean_idx:
            corrupted_cat_txt_list.append(cat_txt)
            corruption_label_list.append(1)
            corruption_type_list.append(None)
            parameter_list.append(None)
        
        elif idx in corrupt_idx:
            atoms, _, _ = str_to_atoms(cat_txt,lat_idx=0)
            corrupted_atoms, corruption_type, parameter = corrupt_atoms(atoms)
            corrupted_cat_txt = atoms_to_str(corrupted_atoms, tagged=False)
            
            corrupted_cat_txt_list.append(corrupted_cat_txt)
            corruption_label_list.append(0)
            corruption_type_list.append(corruption_type)
            parameter_list.append(parameter)
        
    
    return corrupted_cat_txt_list, corruption_label_list, corruption_type_list, parameter_list


def convert(args):
    """
    Convert data to dataframe with string representation
    
    to-do: add process to generate invalid catalyst data
    """

    id_list = []
    atoms_list = []
    ads_list = None
    get_energy = False
    energy_list = []
    prop_list = []
    props = args.props
    
    # convert lmdb to ase atoms
    if args.data_type == 'lmdb':
        ads_list = []
        
        with open(abs_path + '/data/mapping/oc20_data_mapping_light.pkl','rb') as fr:
            mapping = pickle.load(fr)
                    
        dataset = LmdbDataset({'src':args.src_path})
        check_key = True
        key_exists = True
                
        for i in tqdm(range(len(dataset)),desc='Converting'):
            data = dataset[i]
            id_ = data.sid
            
            # adsorbate check
            if check_key:
                if 'random'+str(id_) not in mapping.keys():
                    print(f"Warning: Key 'random{id_}' not found in mapping.")
                    key_exists = False
                    ads = None
                else:
                    ads = mapping['random' + str(id_)]['ads']
                    check_key = False
            elif key_exists:
                ads = mapping['random' + str(id_)]['ads']
            
            # prepare properties
            if props is not None:
                if 'energy' in props:
                    props.remove('energy')
                    get_energy = True
                prop_it = []
                for prop in props:
                    prop_it.append(mapping['random' + str(id_)][prop])
                    
            batch = data_list_collater([data])
            
            # energy check
            if not hasattr(batch,'energy'):
                for att in ['y','y_init']:
                    if hasattr(batch,att):
                        batch.energy = getattr(batch, att)
            
            # append energy
            if get_energy:
                assert hasattr(batch,'energy'), 'There is no energy in dataset'
                energy_list.append(round(batch.energy.item(),2))
            
            # force check
            if ~hasattr(batch,'force') and hasattr(batch,'forces'):
                batch.force = batch.forces
                
            atoms = batch_to_atoms(batch)
            
            if hasattr(data,'fid'):
                id_ = str(id_) + str(data.fid)
                
            id_list.append(id_)
            atoms_list.extend(atoms)
            ads_list.append(ads)
            prop_list.append(prop_it)
            
    elif args.data_type == 'ase':
        if os.path.isdir(arg.src_path):
            atoms_list = [read(arg.src_path + i) for i in os.listdir(arg.src_path)]
        else:
            atoms_list = read(arg.src_path, ':')
        id_list = list(range(len(atoms_list)))
    
    # convert atoms to string
    is_tagged =  2 in atoms_list[0].get_tags()
    cat_txt_list = [atoms_to_str(atoms,tagged=is_tagged) for atoms in tqdm(atoms_list,desc='Atoms to string')]
    
    df = pd.DataFrame(columns = ['id','cat_txt'])
    df['id'] = id_list
    df['cat_txt'] = cat_txt_list
    
    if ads_list is not None:
        df['ads_symbol'] = ads_list
    
    if props is not None:
        prop_list = [list(x) for x in zip(*prop_list)]
        for idx, prop in enumerate(props):
            df[prop] = prop_list[idx]
    
    if get_energy:
        df['energy'] = energy_list
        
    
    if args.corrupt_data:
        corrupted_cat_txt_list, corruption_label_list, corruption_type_list, parameter_list = corrupt_data(cat_txt_list)
        df['corrupted_cat_txt'] = corrupted_cat_txt_list
        df['corruption_label'] = corruption_label_list
        df['corruption_type'] =  corruption_type_list
        df['parameter'] = parameter_list
    
    df.to_csv(os.path.join(args.dst_path, args.name + '.csv'))

if __name__ == '__main__':
    args = parse_args()
    convert(args) 

    
    

    
        
        
        
        
        
        
        
            
