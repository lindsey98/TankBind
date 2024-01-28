from Bio.PDB.PDBList import PDBList  # pip install biopython if import failure
import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from tankbind.feature_utils import split_protein_and_ligand, generate_sdf_from_smiles_using_rdkit, get_protein_feature, extract_torchdrug_feature_from_mol
from tankbind.generation_utils import get_LAS_distance_constraint_mask, get_info_pred_distance, write_with_new_coords

import rdkit.Chem as Chem

from rdkit import Chem
from rdkit.Chem import Draw
from tankbind.data import TankBind_prediction
import logging
from torch_geometric.loader import DataLoader
from tqdm import tqdm    # pip install tqdm if fails.
from tankbind.model import get_model
import torch

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

if __name__ == '__main__':

    # old = torch.load('./protein.pt')
    # new = torch.load('./examples/single_PDB_6hd6/6hd6_dataset/processed/protein.pt')

    pre = "./examples/single_PDB_6hd6/"
    pdir = f"{pre}/PDBs/"

    pdb = '6hd6'
    pdbl = PDBList()
    native_pdb = pdbl.retrieve_pdb_file(pdb, pdir=pdir, file_format='pdb')


    # Load your molecule (example with RDKit)
    # molecule = Chem.MolFromPDBFile(f"{pre}/{pdb}_protein.pdb")
    # img = Draw.MolToImage(molecule)
    # img.save(f"{pre}/output_image.png")

    parser = PDBParser(QUIET=True)
    s = parser.get_structure(pdb, native_pdb)
    print(list(s.get_chains()))

    # list HETATMs
    c = s[0]['A']
    res_list = list(c.get_residues())
    hetro_list = [res for res in res_list if (res.full_id[-1][0] != ' ' and res.full_id[-1][0] != 'W')]
    print(hetro_list)

    # save cleaned protein.
    ligand_seq_id = 602
    proteinFile = f"{pre}/{pdb}_protein.pdb"
    ligandFile = f"{pre}/{pdb}_FYH_ligand.sdf"
    clean_res_list, ligand_list = split_protein_and_ligand(c, pdb, ligand_seq_id, proteinFile, ligandFile)

    ligand_seq_id = 603
    ligandFile = f"{pre}/{pdb}_STI_ligand.sdf"
    clean_res_list, ligand_list = split_protein_and_ligand(c, pdb, ligand_seq_id, proteinFile, ligandFile)

    # we also generated a sdf file using RDKit.
    ligandFile = f"{pre}/{pdb}_STI_ligand.sdf"
    smiles = Chem.MolToSmiles(Chem.MolFromMolFile(ligandFile))
    rdkitMolFile = f"{pre}/{pdb}_STI_mol_from_rdkit.sdf"
    shift_dis = 0  # for visual only, could be any number, shift the ligand away from the protein.
    generate_sdf_from_smiles_using_rdkit(smiles, rdkitMolFile, shift_dis=shift_dis)

    # we also generated a sdf file using RDKit.
    ligandFile = f"{pre}/{pdb}_FYH_ligand.sdf"
    smiles = Chem.MolToSmiles(Chem.MolFromMolFile(ligandFile))
    rdkitMolFile = f"{pre}/{pdb}_FYH_mol_from_rdkit.sdf"
    shift_dis = 0  # for visual only, could be any number, shift the ligand away from the protein.
    generate_sdf_from_smiles_using_rdkit(smiles, rdkitMolFile, shift_dis=shift_dis)

    '''Protein features'''
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("x", proteinFile)
    res_list = list(s.get_residues())
    protein_dict = {}
    protein_dict[pdb] = get_protein_feature(res_list)
    #
    # '''ligand features'''
    compound_dict = {}
    for ligandName in ['STI', 'FYH']:
        rdkitMolFile = f"{pre}/{pdb}_{ligandName}_mol_from_rdkit.sdf"
        mol = Chem.MolFromMolFile(rdkitMolFile)
        compound_dict[pdb + f"_{ligandName}" + "_rdkit"] = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)

    '''protein pockets features'''
    pdb_list = [pdb]
    ds = f"{pre}/protein_list.ds"
    with open(ds, "w") as out:
        for pdb in pdb_list:
            out.write(f"./{pdb}_protein.pdb\n")

    # p2rank = "bash ./p2rank_2.4.1/prank"
    p2rank = "bash ./p2rank_2.3/prank"
    cmd = f"{p2rank} predict {ds} -o {pre}/p2rank -threads 1"
    os.system(cmd)
    #
    # '''Aggregate all features together'''
    info = []
    for pdb in pdb_list:
        for compound_name in list(compound_dict.keys()):
            # use protein center as the block center.
            com = ",".join([str(a.round(3)) for a in protein_dict[pdb][0].mean(axis=0).numpy()])
            info.append([pdb, compound_name, "protein_center", com])

            p2rankFile = f"{pre}/p2rank/{pdb}_protein.pdb_predictions.csv"
            pocket = pd.read_csv(p2rankFile)
            pocket.columns = pocket.columns.str.strip()
            pocket_coms = pocket[['center_x', 'center_y', 'center_z']].values
            for ith_pocket, com in enumerate(pocket_coms):
                com = ",".join([str(a.round(3)) for a in com])
                info.append([pdb, compound_name, f"pocket_{ith_pocket + 1}", com])
    info = pd.DataFrame(info, columns=['protein_name', 'compound_name', 'pocket_name', 'pocket_com'])
    print(info)

    dataset_path = f"{pre}/{pdb}_dataset/"
    os.system(f"rm -r {dataset_path}")
    os.system(f"mkdir -p {dataset_path}")
    dataset = TankBind_prediction(dataset_path, data=info, protein_dict=protein_dict, compound_dict=compound_dict)
    #

    # '''Predict'''
    dataset_path = f"{pre}/{pdb}_dataset/"
    dataset = TankBind_prediction(dataset_path)

    batch_size = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    logging.basicConfig(level=logging.INFO)
    model = get_model(0, logging, device)
    # re-dock model
    # modelFile = "./saved_models/re_dock.pt"
    # self-dock model
    modelFile = "./saved_models/self_dock.pt"
    model.load_state_dict(torch.load(modelFile, map_location=device))
    _ = model.eval()

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             follow_batch=['x', 'y', 'compound_pair'],
                             shuffle=False,
                             num_workers=8)
    affinity_pred_list = []
    y_pred_list = []
    for data in tqdm(data_loader):
        data = data.to(device)
        y_pred, affinity_pred = model(data)
        affinity_pred_list.append(affinity_pred.detach().cpu())
        for i in range(data.y_batch.max() + 1):
            y_pred_list.append((y_pred[data['y_batch'] == i]).detach().cpu())

    affinity_pred_list = torch.cat(affinity_pred_list)
    info = dataset.data
    info['affinity'] = affinity_pred_list
    info.to_csv(f"{pre}/info_with_predicted_affinity.csv")

    chosen = info.loc[
        info.groupby(['protein_name', 'compound_name'], sort=False)['affinity'].agg('idxmax')].reset_index()
    print(chosen)

    for i, line in chosen.iterrows():
        idx = line['index']
        pocket_name = line['pocket_name']
        compound_name = line['compound_name']
        ligandName = compound_name.split("_")[1]

        coords = dataset[idx].coords.to(device)
        protein_nodes_xyz = dataset[idx].node_xyz.to(device)
        n_compound = coords.shape[0]
        n_protein = protein_nodes_xyz.shape[0]

        y_pred = y_pred_list[idx].reshape(n_protein, n_compound).to(device)
        y = dataset[idx].dis_map.reshape(n_protein, n_compound).to(device)
        compound_pair_dis_constraint = torch.cdist(coords, coords)

        rdkitMolFile = f"{pre}/{pdb}_{ligandName}_mol_from_rdkit.sdf"
        mol = Chem.MolFromMolFile(rdkitMolFile)
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol).bool()
        LAS_distance_constraint_mask = LAS_distance_constraint_mask.to(device)

        info = get_info_pred_distance(coords,
                                      y_pred,
                                      protein_nodes_xyz,
                                      compound_pair_dis_constraint,
                                      LAS_distance_constraint_mask=LAS_distance_constraint_mask,
                                      n_repeat=1,
                                      show_progress=False)

        result_folder = f'{pre}/{pdb}_result/'
        os.makedirs(result_folder, exist_ok=True)
        # toFile = f'{result_folder}/{ligandName}_{pocket_name}_tankbind.sdf'
        toFile = f'{result_folder}/{ligandName}_tankbind.sdf'
        # print(toFile)
        new_coords = info.sort_values("loss")['coords'].iloc[0].astype(np.double)
        write_with_new_coords(mol, new_coords, toFile)