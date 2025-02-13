import sys, os
abs_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(abs_path)

import argparse
import pandas as pd
import numpy as np
import pickle
import random
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
    parser.add_argument('--num-workers',
                        type=int,
                        default=1,
                        help='no. of processes',    
                        )
    parser.add_argument('--props',
                        nargs='+',
                        choices=['ads','spg','miller','energy','comp'],
                        help="list of properties to include dataframe, separated by spaces (e.g., --props spg mil energy)"
                        )
    args = parser.parse_args()
    return args

g_mapping = None
g_props = None
g_get_energy = False
g_dataset = None


def init_worker(args, mapping, props, get_energy):
    global g_mapping, g_props, g_get_energy, g_dataset
    g_mapping = mapping
    g_props = props
    g_get_energy = get_energy
    g_dataset = LmdbDataset({'src': args.src_path})

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

def process_item(i):
    global g_mapping, g_props, g_get_energy, g_dataset
    data = g_dataset[i]
    id_ = data.sid
    key = 'random' + str(id_)
    
    #if key in g_mapping:
    #    ads = g_mapping[key]['ads']
    #else:
    #    ads = None
    #    print(f"Warning: Key '{key}' not found in mapping.")
    
    prop_it = []
    if g_props is not None:
        for prop in g_props:
            prop_it.append(g_mapping.get(key, {}).get(prop, None))
    
    batch = data_list_collater([data])
    
    energy_att = 'energy'
    if not hasattr(batch, energy_att):
        for att in ['y_init','y']:
            if hasattr(batch, att):
                energy_att = att
                break
                
    energy_str = None
    if g_get_energy:
        if not hasattr(batch, energy_att):
            raise AssertionError('There is no energy in dataset')
        energy = getattr(batch,energy_att).item()
        energy_val = max(-10,min(round(energy/2,2)*2,10))
        energy_str = f'{energy_val:.1f}'
    
    if not hasattr(batch, 'force') and hasattr(batch, 'forces'):
        batch.force = batch.forces
    
    atoms = batch_to_atoms(batch)
    
    if hasattr(data, 'fid'):
        id_ = str(id_) + '_' + str(data.fid)
    
    return (id_, atoms, prop_it, energy_str)

def process_atoms_to_str(args):
    atoms, tagged = args
    return atoms_to_str(atoms, tagged=tagged)
    
def convert(args):
    """
    Convert data to dataframe with string representation
    
    to-do: add process to generate invalid catalyst data
    """

    id_list = []
    atoms_list = []
    #ads_list = None
    energy_list = []
    prop_list = []
    
    local_props = None
    if args.props is not None:
        local_props = list(args.props)
        
    # convert lmdb to ase atoms
    if args.data_type == 'lmdb':
        #ads_list = []
        
        mapping_path = os.path.join(abs_path, 'data/mapping/oc20_data_mapping_light.pkl')
        with open(mapping_path, 'rb') as fr:
            mapping = pickle.load(fr)
                    
        dataset_temp = LmdbDataset({'src':args.src_path})
        n_items = len(dataset_temp)
        del dataset_temp
        
        # process energy if present
        get_energy = False
        if local_props is not None:
            if 'energy' in local_props:
                get_energy = True
                local_props = [p for p in local_props if p != 'energy']
        
        # set up MP
        pool = mp.Pool(processes=args.num_workers,
               initializer=init_worker,
               initargs=(args, mapping, local_props, get_energy))
               
        # process items
        results = list(tqdm(pool.imap(process_item, range(n_items)),
                            total=n_items,
                            desc="Converting"))
                                
        pool.close()
        pool.join()        
        
        for res in results:
            id_, atoms_res, prop_it, energy_str = res
            id_list.append(id_)
            atoms_list.extend(atoms_res)  # Note: atoms_res is a list.
            #ads_list.append(ads)
            prop_list.append(prop_it)
            if get_energy:
                energy_list.append(energy_str)                
            
    elif args.data_type == 'ase':
        if os.path.isdir(arg.src_path):
            atoms_list = [read(arg.src_path + i) for i in os.listdir(arg.src_path)]
        else:
            atoms_list = read(arg.src_path, ':')
        id_list = list(range(len(atoms_list)))
    
    # convert atoms to string
    is_tagged =  2 in atoms_list[0].get_tags()
    atoms_args = [(atoms, is_tagged) for atoms in atoms_list]
    
    with mp.Pool(processes=args.num_workers) as pool:
        cat_txt_list = list(tqdm(pool.imap(process_atoms_to_str, atoms_args),
                                 total=len(atoms_args),
                                 desc="Converting atoms to string"))
    
    df = pd.DataFrame(columns = ['id','cat_txt'])
    df['id'] = id_list
    df['cat_txt'] = cat_txt_list
    
    #if ads_list is not None:
    #    df['ads_symbol'] = ads_list
    
    # add props
    if local_props is not None and len(prop_list) > 0:
        prop_list_transposed = [list(x) for x in zip(*prop_list)]
        for idx, prop in enumerate(local_props):
            df[prop] = prop_list_transposed[idx]
    
    if args.data_type == 'lmdb' and local_props is not None and get_energy:
        df['energy'] = energy_list
        
    if args.corrupt_data:
        corrupted_cat_txt_list, corruption_label_list, corruption_type_list, parameter_list = corrupt_data(cat_txt_list)
        df['corrupted_cat_txt'] = corrupted_cat_txt_list
        df['corruption_label'] = corruption_label_list
        df['corruption_type'] =  corruption_type_list
        df['parameter'] = parameter_list
    
    output_csv = os.path.join(args.dst_path, args.name + '.csv')
    df.to_csv(output_csv, index=True)
    print(f"Data saved to {output_csv}")
    
if __name__ == '__main__':
    args = parse_args()
    convert(args) 

    
    

    
        
        
        
        
        
        
        
            
