import pandas as pd
from tqdm import tqdm
from tankbind.utils import read_pdbbind_data
from tankbind.feature_utils import read_mol, write_renumbered_sdf, get_protein_feature, select_chain_within_cutoff_to_ligand_v2, get_protein_feature, get_clean_res_list, extract_torchdrug_feature_from_mol
from rdkit import RDLogger
import os
import mlcrate as mlc
import os
from Bio.PDB import PDBParser
import torch
import numpy as np
from functools import partial
from tankbind.data import TankBindDataSet

def batch_get_protein_feature(x):
    protein_dict = {}
    pdb, proteinFile, toFile = x
    parser = PDBParser(QUIET=True)
    s = parser.get_structure(pdb, proteinFile)
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    protein_dict[pdb] = get_protein_feature(res_list)
    torch.save(protein_dict, toFile)

def assign_group(pdb, valid, test):
    if pdb in valid:
        return 'valid'
    if pdb in test:
        return 'test'
    return 'train'

if __name__ == '__main__':

    pre = "./datasets"
    tankbind_data_path = f"{pre}/tankbind_data"
    protein_embedding_folder = f"{tankbind_data_path}/gvp_protein_embedding"
    p2rank_prediction_folder = f"{pre}/p2rank_protein_remove_extra_chains_10A"
    os.makedirs(tankbind_data_path, exist_ok=True)

    df_pdb_id = pd.read_csv(f'{pre}/index/INDEX_general_PL_name.2020', sep="  ", comment='#',
                            header=None, names=['pdb', 'year', 'uid', 'd', 'e','f','g','h','i','j','k','l','m','n','o'],
                            engine='python')
    df_pdb_id = df_pdb_id[['pdb','uid']]

    data = read_pdbbind_data(f'{pre}/index/INDEX_general_PL_data.2020')
    data = data.merge(df_pdb_id, on=['pdb'])

    '''Remove ligands that are unreadable by RDKit'''
    RDLogger.DisableLog('rdApp.*')
    pdb_list = []
    error_list = []
    for pdb in tqdm(data.pdb):
        sdf_fileName = f"{pre}/merged_pdbbind_files/{pdb}/{pdb}_ligand.sdf" # ligand
        mol2_fileName = f"{pre}/merged_pdbbind_files/{pdb}/{pdb}_ligand.mol2"
        mol, error = read_mol(sdf_fileName, mol2_fileName)
        if error:
            error_list.append(pdb)
            continue
        pdb_list.append(pdb)

    data = data.query("pdb in @pdb_list").reset_index(drop=True) # where the values in the 'pdb' column are present in the list pdb_list.
    exclude_pdb = ['2r1w', '5ujo', '3kqs', '4xkc']  # List of PDB codes to exclude
    data = data.query("pdb not in @exclude_pdb").reset_index(drop=True)
    print('Valid pairs: ', data.shape) # fixme: I find different rdkit version will affect num valid pairs

    '''Write the readable ligands, and for ease of RMSD evaluation later, we renumber the atom index to be consistent with the smiles'''
    # toFolder = f"{pre}/renumber_atom_index_same_as_smiles"
    # os.makedirs(toFolder, exist_ok=True)
    # for pdb in tqdm(pdb_list):
    #     sdf_fileName = f"{pre}/merged_pdbbind_files/{pdb}/{pdb}_ligand.sdf"
    #     mol2_fileName = f"{pre}/merged_pdbbind_files/{pdb}/{pdb}_ligand.mol2"
    #     toFile = f"{toFolder}/{pdb}.sdf"
    #     write_renumbered_sdf(toFile, sdf_fileName, mol2_fileName)

    '''We also reduced the possibility of encountering equally valid binding sites by removing chains that have no atom within 10Å from any atom of the ligand following the protocol described in [36].'''
    # toFolder = f"{pre}/protein_remove_extra_chains_10A/"
    # os.makedirs(toFolder, exist_ok=True)
    # input_ = []
    # cutoff = 10
    # for pdb in data.pdb.values:
    #     pdbFile = f"{pre}/merged_pdbbind_files/{pdb}/{pdb}_protein.pdb"
    #     ligandFile = f"{pre}/renumber_atom_index_same_as_smiles/{pdb}.sdf"
    #     toFile = f"{toFolder}/{pdb}_protein.pdb"
    #     x = (pdbFile, ligandFile, cutoff, toFile)
    #     input_.append(x)
    #
    # pool = mlc.SuperPool(64)
    # pool.pool.restart()
    # _ = pool.map(select_chain_within_cutoff_to_ligand_v2, input_)
    # pool.exit()

    '''Segment the protein into functional blocks with p2rank segmentation. First download p2rank'''
    # ds = f"protein_list.ds"
    # with open(ds, "w") as out:
    #     for pdb in data.pdb.values:
    #         if os.path.exists(f"/home/ruofan/git_space/TankBind/datasets/protein_remove_extra_chains_10A/{pdb}_protein.pdb"):
    #             out.write(f"/home/ruofan/git_space/TankBind/datasets/protein_remove_extra_chains_10A/{pdb}_protein.pdb\n")
    #
    # # takes about 30 minutes.
    # os.makedirs(p2rank_prediction_folder, exist_ok=True)
    # p2rank = "bash ./p2rank_2.4.1/prank"
    # cmd = f"{p2rank} predict {ds} -threads 16 -o {p2rank_prediction_folder}"
    # os.system(cmd)

    # data.to_csv(f"{pre}/data.csv")
    # pdb_list = data.pdb.values
    # name_list = pdb_list
    # d_list = []

    #
    # for name in tqdm(name_list):
    #     p2rankFile = f"{pre}/p2rank_protein_remove_extra_chains_10A/{name}_protein.pdb_predictions.csv"
    #     d = pd.read_csv(p2rankFile)
    #     d.columns = d.columns.str.strip()
    #     d_list.append(d.assign(name=name))
    # d = pd.concat(d_list).reset_index(drop=True)
    # d.reset_index(drop=True).to_feather(f"{tankbind_data_path}/p2rank_result.feather")

    '''Embed protein into features'''
    # input_ = []
    # os.makedirs(protein_embedding_folder, exist_ok=True)
    # for pdb in pdb_list:
    #     proteinFile = f"{pre}/protein_remove_extra_chains_10A/{pdb}_protein.pdb"
    #     toFile = f"{protein_embedding_folder}/{pdb}.pt"
    #     x = (pdb, proteinFile, toFile)
    #     input_.append(x)
    #
    # pool = mlc.SuperPool(64)
    # pool.pool.restart()
    # _ = pool.map(batch_get_protein_feature, input_)
    # pool.exit()

    '''Embed ligand into features'''
    # compound_dict = {}
    # skip_pdb_list = []
    # for pdb in tqdm(pdb_list):
    #     mol, _ = read_mol(f"{pre}/renumber_atom_index_same_as_smiles/{pdb}.sdf", None)
    #     # extract features from sdf.
    #     try:
    #         compound_dict[pdb] = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)  # self-dock set has_LAS_mask to true
    #     except Exception as e:
    #         print(e)
    #         skip_pdb_list.append(pdb)
    #         print(pdb)
    # torch.save(compound_dict, f"{tankbind_data_path}/compound_torchdrug_features.pt")

    '''Construct dataset'''
    ### Protein bindng pockets
    d = pd.read_feather(f"{tankbind_data_path}/p2rank_result.feather")
    pockets_dict = {}
    pdb_list = data.pdb.values

    for name in tqdm(pdb_list):
        pockets_dict[name] = d[d.name == name].reset_index(drop=True)

    ### Protein embeddings
    protein_dict = {}
    for pdb in tqdm(pdb_list):
        protein_dict.update(torch.load(f"{protein_embedding_folder}/{pdb}.pt"))

    ### Ligand embeddings
    compound_dict = torch.load(f"{tankbind_data_path}/compound_torchdrug_features.pt")

    valid = np.loadtxt("../EquiBind/data/timesplit_no_lig_overlap_val", dtype=str)
    test = np.loadtxt("../EquiBind/data/timesplit_test", dtype=str)

    data['group'] = data.pdb.map(partial(assign_group,  valid=valid, test=test))
    data['name'] = data['pdb']

    info = []
    for i, line in tqdm(data.iterrows(), total=data.shape[0]):
        pdb = line['pdb']
        uid = line['uid']
        # smiles = line['smiles']
        smiles = ""
        affinity = line['affinity']
        group = line['group']

        compound_name = line['name']
        protein_name = line['name']

        pocket = pockets_dict[pdb].head(10) # top-10 potential binding pockets
        pocket.columns = pocket.columns.str.strip()
        pocket_coms = pocket[['center_x', 'center_y', 'center_z']].values
        # native block.
        info.append([protein_name, compound_name, pdb, smiles, affinity, uid, None, True, False, group])
        # protein center as a block: For some small proteins, no binding site is identified by p2k, we therefore add an extra protein block located at the center of the whole protein
        protein_com = protein_dict[protein_name][0].numpy().mean(axis=0).astype(float).reshape(1, 3)
        info.append([protein_name, compound_name, pdb + "_c", smiles, affinity, uid, protein_com, False, False, group])

        for idx, pocket_line in pocket.iterrows():
            pdb_idx = f"{pdb}_{idx}"
            info.append(
                [protein_name, compound_name, pdb_idx, smiles, affinity, uid, pocket_coms[idx].reshape(1, 3), False,
                 False, group])
    info = pd.DataFrame(info, columns=['protein_name', 'compound_name', 'pdb', 'smiles', 'affinity', 'uid', 'pocket_com',
                                 'use_compound_com', 'use_whole_protein',
                                 'group'])

    toFilePre = f"{pre}/dataset"
    os.makedirs(toFilePre, exist_ok=True)
    dataset = TankBindDataSet(toFilePre, data=info, protein_dict=protein_dict, compound_dict=compound_dict)

    t = []
    data = dataset.data
    pre_pdb = None
    for i, line in tqdm(data.iterrows(), total=data.shape[0]):
        pdb = line['compound_name']
        d = dataset[i]
        p_length = d['node_xyz'].shape[0]
        c_length = d['coords'].shape[0]
        y_length = d['y'].shape[0]
        num_contact = (d.y > 0).sum()
        t.append([i, pdb, p_length, c_length, y_length, num_contact])

    t = pd.DataFrame(t, columns=['index', 'pdb', 'p_length', 'c_length', 'y_length', 'num_contact'])
    t['num_contact'] = t['num_contact'].apply(lambda x: x.item())
    data = pd.concat([data, t[['p_length', 'c_length', 'y_length', 'num_contact']]], axis=1)
    native_num_contact = data.query("use_compound_com").set_index("protein_name")['num_contact'].to_dict()
    data['native_num_contact'] = data.protein_name.map(native_num_contact)
    torch.save(data, f"{toFilePre}/processed/data.pt")

    '''Split test dataset out'''
    info = torch.load(f"{toFilePre}/processed/data.pt")
    test = info.query("group == 'test'").reset_index(drop=True)
    test_pdb_list = info.query("group == 'test'").protein_name.unique()
    subset_protein_dict = {}
    for pdb in tqdm(test_pdb_list):
        subset_protein_dict[pdb] = protein_dict[pdb]
    subset_compound_dict = {}
    for pdb in tqdm(test_pdb_list):
        subset_compound_dict[pdb] = compound_dict[pdb]
    toFilePre = f"{pre}/test_dataset"
    os.makedirs(toFilePre, exist_ok=True)
    dataset = TankBindDataSet(toFilePre,
                              data=test,
                              protein_dict=subset_protein_dict,
                              compound_dict=subset_compound_dict)
    print()