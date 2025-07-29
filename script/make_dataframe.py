import sys, os
abs_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(abs_path)

os.environ["OMP_NUM_THREADS"] = '1'  
os.environ["MKL_NUM_THREADS"] = '1'  

import argparse
import pandas as pd
import numpy as np
import pickle
import random
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
logging.getLogger().setLevel(logging.ERROR)

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
    parser.add_argument('--add-sep', action='store_true', help='whether to add seperation tokens', default=False)
    parser.add_argument('--use-relaxed',
                        action='store_ture', 
                        help='whether to use relaxed sturctures when IS2RE',
                        default=False
                        ) 
    #parser.add_argument('--corrupt-data', 
    #                    action='store_true', 
    #                    help='whether to add corrupted data for training detectino model', 
    #                    default=False
    #                    )  
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
    parser.add_argument('--chunk-size',
                        type=int,
                        default=1000000,
                        help='size of chunk',
                        )
    args = parser.parse_args()
    return args

g_mapping = None
g_props = None
g_get_energy = False
g_use_realxed = False
g_dataset = None


def init_worker(src, mapping, props, get_energy, use_relaxed):
    global g_mapping, g_props, g_get_energy, g_dataset
    g_mapping = mapping
    g_props = props
    g_get_energy = get_energy
    g_use_relaxed = use_relaxed
    g_dataset = LmdbDataset({'src': src})

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
    global g_mapping, g_props, g_get_energy, g_dataset, g_use_relaxed
    data = g_dataset[i]
    id_ = data.sid
    key = 'random' + str(id_)
    
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
        batch.energy = getattr(batch,energy_att)        
        
    energy_str = None
    if g_get_energy:
        if not hasattr(batch, energy_att):
            raise AssertionError('There is no energy in dataset')
        energy = getattr(batch,energy_att).item()
        energy_val = max(-10,min(round(energy/2,2)*2,10))
        energy_str = f'{energy_val:.1f}'
    
    if not hasattr(batch, 'force') and hasattr(batch, 'forces'):
        batch.force = batch.forces
    
    # use relaxed position if possible
    if hasattr(batch, 'pos_relaxed') and g_use_relaxed:
        batch.pos = batch.pos_relaxed
    
    atoms = batch_to_atoms(batch)
    
    if hasattr(data, 'fid'):
        id_ = str(id_) + '_' + str(data.fid)
    
    return (id_, atoms, prop_it, energy_str)

def process_atoms_to_str(args):
    atoms, tagged, add_sep = args
    return atoms_to_str(atoms, tagged=tagged, add_sep=add_sep)
    
def convert(args):
    """
    Convert data to dataframe with string representation
    
    to-do: add process to generate invalid catalyst data
    """
    
    output_csv = os.path.join(args.dst_path, args.name + '.csv')
    local_props = None
    if args.props is not None:
        local_props = list(args.props)
        
    # convert lmdb to ase atoms
    if args.data_type == 'lmdb':
        
        # initialize properties map
        mapping_path = os.path.join(abs_path, 'data/mapping/oc20_data_mapping_light.pkl')
        with open(mapping_path, 'rb') as fr:
            mapping = pickle.load(fr)
        
        # process energy if present
        get_energy = False
        if local_props is not None:
            if 'energy' in local_props:
                get_energy = True
                local_props = [p for p in local_props if p != 'energy']
        
        
        if os.path.isdir(args.src_path):
            src_path = [os.path.join(args.src_path + i) for i in os.listdir(args.src_path) if i.endswith('.lmdb')]
        else:
            src_path = [args.src_path]
            
        for idx, src_chunk in enumerate(src_path):
            dataset_temp = LmdbDataset({'src':src_chunk})
            n_items = len(dataset_temp)
            del dataset_temp
            
            # set up MP
            pool = mp.Pool(processes=args.num_workers,
               initializer=init_worker,
               initargs=(src_chunk, mapping, local_props, get_energy, args.use_relaxed))

               
            # process items
            results = list(tqdm(pool.imap(process_item, range(n_items)),
                                total=n_items,
                                desc=f'Converting lmdb chunk-{idx+1}'))
                                
            pool.close()
            pool.join()        
            
            chunk_size = args.chunk_size
            n_chunks = n_items // chunk_size
            if n_chunks == 0:
                n_chunks_range = [0]
            else:
                n_chunks_range = range(n_chunks)
            
            for chunk_idx in n_chunks_range:
                start_idx = chunk_idx*chunk_size
                end_idx = (chunk_idx+1)*chunk_size                
                ids = []
                atoms_list = []
                props = []
                energies = []

                for res in results[start_idx:end_idx]:
                    id_, atoms_res, prop_it, energy_str = res
                    ids.append(id_)
                    atoms_list.extend(atoms_res)
                    props.append(prop_it)
                    if get_energy:
                        energies.append(energy_str)
                
                is_tagged = 2 in atoms_list[0].get_tags()
                
                atoms_args = [(atoms, is_tagged, args.add_sep) for atoms in atoms_list]
                
                with mp.Pool(processes=args.num_workers) as pool:
                    cat_txt_list = list(tqdm(pool.imap(process_atoms_to_str, atoms_args),
                                             total=len(atoms_args),
                                             desc=f'Converting atoms chunk-{chunk_idx+1}'))
                                             
                df_chunk = pd.DataFrame({
                    'id' : ids,
                    'cat_txt': cat_txt_list
                })
                            
                if local_props and len(props) > 0:
                    props_transposed = [list(x) for x in zip(*props)]
                    for prop_name, prop in zip(local_props, props_transposed):
                        df_chunk[prop_name] = prop
                        
                if get_energy:
                    df_chunk['energy'] = energies
                
                if (chunk_idx==0) & (idx==0):
                    df_chunk.to_csv(output_csv, index=False, mode='w', header=True)
                else:
                    df_chunk.to_csv(output_csv, index=False, mode='a', header=False)
                    
                del ids, atoms_list, props, energies, cat_txt_list, atoms_args
            
            if n_chunks != 0:    
                ids = []
                atoms_list = []
                props = []
                energies = []
                
                for res in results[end_idx:]:
                    id_, atoms_res, prop_it, energy_str = res
                    ids.append(id_)
                    atoms_list.extend(atoms_res)
                    props.append(prop_it)
                    if get_energy:
                        energies.append(energy_str)
                
                is_tagged = 2 in atoms_list[0].get_tags()
                atoms_args = [(atoms, is_tagged, args.add_sep) for atoms in atoms_list]
                
                with mp.Pool(processes=args.num_workers) as pool:
                    cat_txt_list = list(tqdm(pool.imap(process_atoms_to_str, atoms_args),
                                             total=len(atoms_args),
                                             desc=f'Converting last atoms chunk'))
                                             
                df_chunk = pd.DataFrame({
                    'id' : ids,
                    'cat_txt': cat_txt_list
                })
                
                if local_props and len(props) > 0:
                    props_transposed = [list(x) for x in zip(*props)]
                    for prop_name, prop in zip(local_props, props_transposed):
                        df_chunk[prop_name] = prop
                        
                if get_energy:
                    df_chunk['energy'] = energies
                    
                df_chunk.to_csv(output_csv, index=False, mode='a', header=False)
       
            
    elif args.data_type == 'ase':
        if os.path.isdir(arg.src_path):
            atoms_list = [read(arg.src_path + i) for i in os.listdir(arg.src_path)]
        else:
            atoms_list = read(arg.src_path, ':')
        ids = list(range(len(atoms_list)))
        
        is_tagged = 2 in atoms_list[0].get_tags()
        atoms_args = [(atoms, is_tagged, args.add_sep) for atoms in atoms_list]
            
        with mp.Pool(processes=args.num_workers) as pool:
            cat_txt_list = list(tqdm(pool.imap(process_atoms_to_str, atoms_args),
                                     total=len(atoms_args),
                                     desc=f'Converting atoms to string'))
        
        df = pd.DataFrame({
            'id' : ids,
            'cat_txt' : cat_txt_list
        })
        
        df.to_csv(output_csv, index=False)
    
    """
    # temporary removed    
    if args.corrupt_data:
        corrupted_cat_txt_list, corruption_label_list, corruption_type_list, parameter_list = corrupt_data(cat_txt_list)
        df['corrupted_cat_txt'] = corrupted_cat_txt_list
        df['corruption_label'] = corruption_label_list
        df['corruption_type'] =  corruption_type_list
        df['parameter'] = parameter_list
    """
    print(f"Data saved to {output_csv}")
    
if __name__ == '__main__':
    args = parse_args()
    convert(args) 

    
    

    
        
        
        
        
        
        
        
            
