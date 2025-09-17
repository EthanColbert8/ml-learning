import os
from tqdm import tqdm
import numpy as np
import awkward as ak
import vector
import torch
# from datasets import load_dataset

def PadJets(input_arrays, max_jets=8):
    """
    Adds zero-padding to awkward array of jets to make it a regular array.
    """
    jets = input_arrays['jets']
    jets_btag = input_arrays['jets_btag']

    input_jets = ak.concatenate([jets.px[:,:,None], jets.py[:,:,None], jets.pz[:,:,None], jets.E[:,:,None], jets_btag[:,:,None]], axis=2)
    input_jets_tensors = [torch.tensor(x) for x in input_jets]

    jets_padded = torch.nested.to_padded_tensor(torch.nested.nested_tensor(input_jets_tensors), padding=0.0)

    jets_padded = jets_padded[:, :max_jets, :]

    jet_pad_mask = torch.all(jets_padded == 0.0, dim=-1)
    others_pad_mask = torch.zeros([len(input_arrays["lepton"]), 3], dtype=torch.bool)
    pad_mask = torch.cat([jet_pad_mask, others_pad_mask], dim=1)

    return jets_padded, pad_mask

def GetLeptonCharge(gen_particles_3d, gen_particles_PID, electrons, muons):
    """
    Because charge of reco-level leptons is not stored in Collide dataset,
    we attempt to infer it by matching to gen-level leptons. This does a
    simplistic delta-R match and infers the charge from that. It is
    VERY UNSAFE and does no checking to address repeated matches or bad
    match quality.
    """
    gen_electron_pid_mask = np.abs(gen_particles_PID) == 11
    gen_muon_pid_mask = np.abs(gen_particles_PID) == 13

    gen_electrons_3d = gen_particles_3d[gen_electron_pid_mask]
    gen_muons_3d = gen_particles_3d[gen_muon_pid_mask]
    gen_electrons_pid = gen_particles_PID[gen_electron_pid_mask]
    gen_muons_pid = gen_particles_PID[gen_muon_pid_mask]

    

    electrons_deltaR = electrons_real[:,:,None].deltaR(gen_electrons_3d[:,None,:])
    muons_deltaR = muons_real[:,:,None].deltaR(gen_muons_3d[:,None,:])

    elecrons_match_idx = ak.argmin(ak.values_astype(electrons_deltaR, np.float32), axis=-1, mask_identity=False)
    muons_match_idx = ak.argmin(ak.values_astype(muons_deltaR, np.float32), axis=-1, mask_identity=False)

    event_level_mask = ~(ak.any(electrons_match_idx == -1, axis=-1) | ak.any(muons_match_idx == -1, axis=-1))

    gen_electrons_pid = gen_electrons_pid[event_level_mask]
    gen_muons_pid = gen_muons_pid[event_level_mask]
    electrons_match_idx = electrons_match_idx[event_level_mask]
    muons_match_idx = muons_match_idx[event_level_mask]

    electrons_charge = -np.sign(gen_electrons_pid[electrons_match_idx])
    muons_charge = -np.sign(gen_muons_pid[muons_match_idx])

    return electrons_charge, muons_charge, event_level_mask

if (__name__ == "__main__"):
    columns_to_load = [
        'FullReco_GenPart_Status',
        'FullReco_GenPart_PT',
        'FullReco_GenPart_Eta',
        'FullReco_GenPart_Phi',
        'FullReco_GenPart_PID',

        'FullReco_Electron_PT',
        'FullReco_Electron_Eta',
        'FullReco_Electron_Phi',
        'FullReco_MuonTight_PT',
        'FullReco_MuonTight_Eta',
        'FullReco_MuonTight_Phi',
        'FullReco_JetPuppiAK4_PT',
        'FullReco_JetPuppiAK4_Eta',
        'FullReco_JetPuppiAK4_Phi',
        'FullReco_JetPuppiAK4_Mass',
        'FullReco_JetPuppiAK4_BTag',
        'FullReco_PUPPIMET_MET',
        'FullReco_PUPPIMET_Phi',
        # 'FullReco_PUPPIMET_Eta',
    ]

    save_dir = "/depot/cms/users/colberte/ml-learning/PyTorch/data"
    data_dir = "/depot/cms/top/jprodger/Bumblebee/src/Bumblebee_Collide/parquet_data/tt0123j_5f_ckm_LO_MLM_leptonic"
    data_files = os.listdir(data_dir)

    print(f"Found {len(data_files)} data files. Processing...")

    lepton_list = []
    lbar_list = []
    gen_top_list = []
    gen_tbar_list = []
    met_list = []
    jets_list = []
    jets_btag_list = []

    for data_file in tqdm(data_files):
        if not data_file.endswith(".parquet"):
            continue
        data_path = os.path.join(data_dir, data_file)
        data_array = ak.from_parquet(data_path, columns=columns_to_load)

        gen_particles_3d = vector.Array(ak.zip({
            "pt": data_array['FullReco_GenPart_PT'],
            "eta": data_array['FullReco_GenPart_Eta'],
            "phi": data_array['FullReco_GenPart_Phi'],
        }))
        gen_particles_PID = data_array['FullReco_GenPart_PID']

        electrons = vector.Array(ak.zip({
            "pt": data_array['FullReco_Electron_PT'],
            "eta": data_array['FullReco_Electron_Eta'],
            "phi": data_array['FullReco_Electron_Phi'],
            "mass": 0.000511 + ak.zeros_like(data_array['FullReco_Electron_PT']),
        }))
        muons = vector.Array(ak.zip({
            "pt": data_array['FullReco_MuonTight_PT'],
            "eta": data_array['FullReco_MuonTight_Eta'],
            "phi": data_array['FullReco_MuonTight_Phi'],
            "mass": 0.10566 + ak.zeros_like(data_array['FullReco_MuonTight_PT']),
        }))

        electrons_charge, muons_charge, event_mask_1 = GetLeptonCharge(gen_particles_3d, gen_particles_PID, electrons, muons)

        # Need to apply the charge-matching mask to everything
        electrons = electrons[event_mask_1]
        muons = muons[event_mask_1]
        data_array = data_array[event_mask_1]
        gen_particles_3d = gen_particles_3d[event_mask_1]
        gen_particles_PID = gen_particles_PID[event_mask_1]

        leptons = ak.concatenate([electrons[electrons_charge < 0], muons[muons_charge < 0]], axis=1)
        lbars = ak.concatenate([electrons[electrons_charge > 0], muons[muons_charge > 0]], axis=1)

        event_mask_2 = ((ak.num(leptons) > 0) & (ak.num(lbars) > 0))

        data_array = data_array[event_mask_2]
        gen_particles_3d = gen_particles_3d[event_mask_2]
        gen_particles_PID = gen_particles_PID[event_mask_2]
        leptons = leptons[event_mask_2]
        lbars = lbars[event_mask_2]

        lepton_cut = leptons.pt == ak.max(ak.values_astype(leptons.pt, np.float32), axis=1, mask_identity=False)
        lbar_cut = lbars.pt == ak.max(ak.values_astype(lbars.pt, np.float32), axis=1, mask_identity=False)
        lepton = vector.MomentumNumpy4D(ak.to_regular(leptons[lepton_cut]).to_numpy().squeeze())
        lbar = vector.MomentumNumpy4D(ak.to_regular(lbars[lbar_cut]).to_numpy().squeeze())

        # Define gen tops, MET, and jets
        gen_tops_mask = ((np.abs(gen_particles_PID) == 6) & (data_array['FullReco_GenPart_Status'] == 62))
        gen_tops_3d = gen_particles_3d[gen_tops_mask]
        gen_tops_pid = gen_particles_PID[gen_tops_mask]

        gen_top_3d = gen_tops_3d[gen_tops_pid > 0]
        gen_top = vector.array({
            "pt": gen_top_3d.pt[:,0],
            "eta": gen_top_3d.eta[:,0],
            "phi": gen_top_3d.phi[:,0],
            "M": 172.5 + np.zeros(len(gen_top_3d))
        })

        gen_tbar_3d = gen_tops_3d[gen_tops_pid < 0]
        gen_tbar = vector.array({
            "pt": gen_tbar_3d.pt[:,0],
            "eta": gen_tbar_3d.eta[:,0],
            "phi": gen_tbar_3d.phi[:,0],
            "M": 172.5 + np.zeros(len(gen_tbar_3d))
        })

        met = vector.array({
            "pt": data_array['FullReco_PUPPIMET_MET'][:,0],
            "phi": data_array['FullReco_PUPPIMET_Phi'][:,0]
        })

        jets = vector.Array(ak.zip({
            "pt": ak.values_astype(data_array['FullReco_JetPuppiAK4_PT'], np.float32),
            "eta": ak.values_astype(data_array['FullReco_JetPuppiAK4_Eta'], np.float32),
            "phi": ak.values_astype(data_array['FullReco_JetPuppiAK4_Phi'], np.float32),
            "mass": ak.values_astype(data_array['FullReco_JetPuppiAK4_Mass'], np.float32),
        }))
        jets_btag = ak.values_astype(data_array['FullReco_JetPuppiAK4_BTag'], np.float32)

        # Append model inputs to lists. This includes adding charge info to leptons.
        lepton_list.append(np.column_stack([lepton.px, lepton.py, lepton.pz, lepton.E, (-1 + np.zeros(len(lepton)))]).astype(np.float32))
        lbar_list.append(np.column_stack([lbar.px, lbar.py, lbar.pz, lbar.E, np.ones(len(lbar))]).astype(np.float32))
        gen_top_list.append(np.column_stack([gen_top.px, gen_top.py, gen_top.pz, gen_top.E]).astype(np.float32))
        gen_tbar_list.append(np.column_stack([gen_tbar.px, gen_tbar.py, gen_tbar.pz, gen_tbar.E]).astype(np.float32))
        met_list.append(np.column_stack([met.px, met.py]).astype(np.float32))
        jets_list.append(jets)
        jets_btag_list.append(jets_btag)
    
    print("Processed data files. Creating full dataset...")

    input_arrays = {
        "lepton": np.concatenate(lepton_list, axis=0),
        "lbar": np.concatenate(lbar_list, axis=0),
        "met": np.concatenate(met_list, axis=0),
        "jets": ak.concatenate(jets_list, axis=0),
        "jets_btag": ak.concatenate(jets_btag_list, axis=0),
    }
    target_arrays = {
        "top": np.concatenate(gen_top_list, axis=0),
        "tbar": np.concatenate(gen_tbar_list, axis=0),
    }

    with torch.no_grad():
        padded_jets, pad_mask = PadJets(input_arrays, max_jets=8)

        leptons_tensor = torch.tensor(np.stack([input_arrays["lepton"], input_arrays["lbar"]], axis=1))
        met_tensor = torch.tensor(np.expand_dims(input_arrays["met"], axis=1))

        targets_tensor = torch.tensor(np.stack([target_arrays["top"], target_arrays["tbar"]], axis=1))

        rand_gen = np.random.default_rng(55378008)
        indices = np.arange(len(targets_tensor))
        rand_gen.shuffle(indices)

        first_cut = int(np.rint(0.7 * len(targets_tensor)))
        second_cut = int(np.rint(0.9 * len(targets_tensor)))

        train_indices = indices[:first_cut]
        valid_indices = indices[first_cut:second_cut]
        test_indices = indices[second_cut:]

        train_tensors = [padded_jets[train_indices], leptons_tensor[train_indices], met_tensor[train_indices], pad_mask[train_indices], targets_tensor[train_indices]]
        valid_tensors = [padded_jets[valid_indices], leptons_tensor[valid_indices], met_tensor[valid_indices], pad_mask[valid_indices], targets_tensor[valid_indices]]
        test_tensors = [padded_jets[test_indices], leptons_tensor[test_indices], met_tensor[test_indices], pad_mask[test_indices], targets_tensor[test_indices]]

        train_dataset = torch.utils.data.TensorDataset(*train_tensors)
        valid_dataset = torch.utils.data.TensorDataset(*valid_tensors)
        test_dataset = torch.utils.data.TensorDataset(*test_tensors)

    print(f"Final dataset sizes: {len(train_dataset)} training, {len(valid_dataset)} validation, {len(test_dataset)} testing. Saving...")


    torch.save(train_dataset, os.path.join(save_dir, "ttbar_dilepton_collide_train.pt"))
    torch.save(valid_dataset, os.path.join(save_dir, "ttbar_dilepton_collide_valid.pt"))
    torch.save(test_dataset, os.path.join(save_dir, "ttbar_dilepton_collide_test.pt"))

    print("Finished.")










    






