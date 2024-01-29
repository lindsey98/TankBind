#### Run virtual screening over drugbank
import shutil

from Bio.PDB import PDBParser
from tankbind.feature_utils import get_clean_res_list, get_protein_feature
import pandas as pd
import os
from tqdm import tqdm
from torch_geometric.data import Dataset
from tankbind.utils import construct_data_from_graph_gvp
import rdkit.Chem as Chem    # conda install rdkit -c rdkit if import failure.
from tankbind.feature_utils import extract_torchdrug_feature_from_mol, get_canonical_smiles, generate_sdf_from_smiles_using_rdkit, write_renumbered_sdf, read_mol
from tankbind.data import TankBind_prediction
import torch
import numpy as np
import logging
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from tankbind.model import get_model
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some inputs.')

    # Add arguments with default values
    parser.add_argument('--proteinDirs', default="./datasets/protein_315", help='Directory for protein files')
    parser.add_argument('--proteinDirsUni', default="./datasets/protein_315", help='Directory for protein files (unfied)')

    parser.add_argument('--ligandDirs', default="./datasets/drugbank", help='Directory for ligand files')

    parser.add_argument('--ligandRdkitDirs', default="./datasets/drugbank_rdkit", help='Where to save the ligand RDKit files')

    parser.add_argument('--protein_features', default='./datasets/protein_315.pt', help='Where to save the Protein features file')
    parser.add_argument('--ligand_features', default="./datasets/drugbank_9k.pt", help='Where to save the Ligand features file')
    parser.add_argument('--ds', default="./datasets/protein_315.ds", help='Where to save the prepared input file for p2rank to scan')
    parser.add_argument('--protein_pockets_p2rank', default="./datasets/protein_315_p2rank",
                        help='Where to save the Protein pockets p2rank predictions')

    parser.add_argument('--dataset_path', default='./datasets/protein315_to_drugbank9k/', help='Where to save the dataset')

    parser.add_argument('--modelFile', default="./saved_models/self_dock.pt", help='Pretrained model file path')

    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')

    # Parse the arguments
    args = parser.parse_args()

    os.makedirs(args.ligandRdkitDirs, exist_ok=True)
    os.makedirs(args.proteinDirsUni, exist_ok=True)
    os.makedirs(f"{args.dataset_path}", exist_ok=True)

    '''Protein features'''
    if os.path.exists(args.protein_features):
        protein_dict = torch.load(args.protein_features)
    else:
        parser = PDBParser(QUIET=True)
        protein_dict = {}

        for proteinName in os.listdir(args.proteinDirs):
            proteinFile = os.path.join(args.proteinDirs, proteinName, "ranked_0.pdb") # fixme: only the highest confidence prediction
            shutil.copyfile(proteinFile, os.path.join(args.proteinDirsUni, proteinName+".pdb"))
            s = parser.get_structure(proteinFile, proteinFile)
            res_list = list(s.get_residues())
            clean_res_list = get_clean_res_list(res_list, ensure_ca_exist=True)
            protein_dict[proteinName] = get_protein_feature(clean_res_list)

        torch.save(protein_dict, args.protein_features)

    '''Protein pockets prediction via p2rank'''
    # # with open(args.ds, "w") as out:
    # #     for proteinName in list(protein_dict.keys()):
    # #         out.write(f"protein_315/{proteinName}.pdb\n")
    # #
    # # cmd = f"bash ./p2rank_2.3/prank predict {args.ds} -o {args.protein_pockets_p2rank} -threads 1" ## fixme: you can increase thread
    # # os.system(cmd)
    # # exit()

    '''Ligand features'''
    if os.path.exists(args.ligand_features):
        compound_dict = torch.load(args.ligand_features)
    else:
        compound_dict = {}

        for ligand in tqdm(os.listdir(args.ligandDirs)):
            ligandName = ligand.split('.sdf')[0]
            ligandFile = os.path.join(args.ligandDirs, ligand)
            mol, error = read_mol(ligandFile, None) # unreadable by rdkit
            if error:
                continue

            toFile = f"{args.ligandRdkitDirs}/{ligandName}_from_rdkit.sdf"

            #### During inference, we convert the sdf file to smiles, then regenerate a low-energy conformation with rdkit and re-save to sdf
            smiles = Chem.MolToSmiles(mol)
            generate_sdf_from_smiles_using_rdkit(smiles, toFile, shift_dis=0)
            mol = Chem.MolFromMolFile(toFile)

            try:
                compound_dict[f"{ligandName}" + "_rdkit"] = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)
            except Exception as e:
                print(ligandName, e)

        torch.save(compound_dict, args.ligand_features)

    #
    '''Construct prediction dataset'''
    for proteinName in list(protein_dict.keys()):
        if os.path.exists(f"{args.dataset_path}/{proteinName}") and len(os.listdir(f"{args.dataset_path}/{proteinName}/processed")) >= 5:
            continue
        info = []

        p2rankFile = f"{args.protein_pockets_p2rank}/{proteinName}.pdb_predictions.csv"
        pocket = pd.read_csv(p2rankFile)
        pocket.columns = pocket.columns.str.strip()
        pocket_coms = pocket[['center_x', 'center_y', 'center_z']].values

        for ligandName in list(compound_dict.keys()):
            # use protein center as the block center.
            com = ",".join([str(a.round(3)) for a in protein_dict[proteinName][0].mean(axis=0).numpy()])
            info.append([proteinName, ligandName, "protein_center", com])

            for ith_pocket, com in enumerate(pocket_coms): # for each protein pocket, create a new record
                com = ",".join([str(a.round(3)) for a in com])
                info.append([proteinName, ligandName, f"pocket_{ith_pocket + 1}", com])

        info = pd.DataFrame(info, columns=['protein_name', 'compound_name', 'pocket_name', 'pocket_com'])
        print(info)

        os.system(f"rm -r {args.dataset_path}/{proteinName}")
        os.system(f"mkdir -p {args.dataset_path}")
        dataset = TankBind_prediction(f"{args.dataset_path}/{proteinName}",
                                      data=info,
                                      protein_dict=protein_dict,
                                      compound_dict=compound_dict) # this will construct graph for both the protein and the ligand

    '''Make predictions'''
    logging.basicConfig(level=logging.INFO)
    model = get_model(0, logging, args.device)
    # Use the self-dock model instead of the re-dock model
    model.load_state_dict(torch.load(args.modelFile, map_location=args.device))
    _ = model.eval()

    for proteinName in list(protein_dict.keys()):

        dataset = TankBind_prediction(f"{args.dataset_path}/{proteinName}")

        data_loader = DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 follow_batch=['x', 'y', 'compound_pair'],
                                 shuffle=False,
                                 num_workers=1)

        affinity_pred_list = []
        y_pred_list = []
        for data in tqdm(data_loader):
            data = data.to(args.device)
            y_pred, affinity_pred = model(data) #
            affinity_pred_list.append(affinity_pred.detach().cpu()) # affinity
            for i in range(data.y_batch.max() + 1):
                y_pred_list.append((y_pred[data['y_batch'] == i]).detach().cpu()) # inter-molecule distance map

        affinity_pred_list = torch.cat(affinity_pred_list)

        info = dataset.data
        info['affinity'] = affinity_pred_list

        info.to_csv(f"./datasets/protein315_to_drugbank9k_{proteinName}_results.csv")

        chosen = info.loc[info.groupby(['protein_name', 'compound_name'], sort=False)['affinity'].agg('idxmax')].reset_index()
        print(chosen)

    # from generation_utils import get_LAS_distance_constraint_mask, get_info_pred_distance, write_with_new_coords
    # # pick one with affinity greater than 7.
    # chosen = info.loc[info.groupby(['protein_name', 'compound_name'], sort=False)['affinity'].agg('idxmax')].reset_index()
    # chosen = chosen.query("affinity > 7").reset_index(drop=True)
    # line = chosen.iloc[0]
    # idx = line['index']
    # one_data = dataset[idx]
    # data_with_batch_info = next(iter(DataLoader(dataset[idx:idx+1],
    #                               batch_size=1,
    #                               follow_batch=['x', 'y', 'compound_pair'],
    #                               shuffle=False,
    #                               num_workers=1)))
    # y_pred, affinity_pred = model(data_with_batch_info)
    #
    # coords = one_data.coords.to(device)
    # protein_nodes_xyz = one_data.node_xyz.to(device)
    # n_compound = coords.shape[0]
    # n_protein = protein_nodes_xyz.shape[0]
    # y_pred = y_pred.reshape(n_protein, n_compound).to(device).detach()
    # y = one_data.dis_map.reshape(n_protein, n_compound).to(device)
    # compound_pair_dis_constraint = torch.cdist(coords, coords)

    # smiles = line['smiles']
    # print(smiles)
    # mol = Chem.MolFromSmiles(smiles)
    # mol.Compute2DCoords()
    # LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol).bool()
    # info = get_info_pred_distance(coords, y_pred, protein_nodes_xyz, compound_pair_dis_constraint,
    #                               LAS_distance_constraint_mask=LAS_distance_constraint_mask,
    #                               n_repeat=1, show_progress=False)
    # toFile = f'{base_pre}/one_tankbind.sdf'
    # new_coords = info.sort_values("loss")['coords'].iloc[0].astype(np.double)
    # write_with_new_coords(mol, new_coords, toFile)
