import os
import sys
import torch
import torch.nn as nn
import numpy as np
import miditoolkit
from miditoolkit.midi.containers import Note, Instrument
import traceback

# ----------------------------------------------------------
# 1. SETUP PATHS
# ----------------------------------------------------------
src_path = os.path.abspath(os.path.join(os.getcwd(), 'Rule_Engine', 'figaro', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.seq2seq import Seq2SeqModule
from vocab import RemiVocab, DescriptionVocab

CHECKPOINT_PATH = os.path.join("Rule_ENgine", "figaro", "checkpoints", "figaro-expert.ckpt")

# ----------------------------------------------------------
# 2. HELPER FUNCTIONS
# ----------------------------------------------------------
def get_instrument_name(program_id):
    lookup = {0: "Acoustic Grand Piano", 40: "Violin", 42: "Cello", 24: "Acoustic Guitar (nylon)", 73: "Flute"}
    return lookup.get(program_id, "Acoustic Grand Piano")

def robust_save_midi(tokens, output_path):
    print(f"💾 Saving MIDI to {output_path}...")
    midi_obj = miditoolkit.midi.parser.MidiFile()
    track = Instrument(program=0, is_drum=False, name="Figaro AI")
    
    TICKS_PER_BAR, TICKS_PER_POS = 1920, 120
    c_bar, c_pos, c_velo, c_dur = 0, 0, 80, 480
    notes = []

    for t in tokens:
        t_str = str(t)
        if "_" not in t_str: continue
        try:
            val_str = t_str.split('_')[-1]
            if not val_str.isdigit(): continue
            val = int(val_str)
            if t_str.startswith("Bar_"): c_bar = val
            elif t_str.startswith("Position_"): c_pos = val
            elif t_str.startswith("Velocity_"): c_velo = val * 4 
            elif t_str.startswith("Duration_"): c_dur = val * TICKS_PER_POS
            elif t_str.startswith("Pitch_") and "drum" not in t_str:
                # Align notes relative to our start at Bar 2
                relative_bar = max(0, c_bar - 2) 
                start = (relative_bar * TICKS_PER_BAR) + (c_pos * TICKS_PER_POS)
                notes.append(Note(velocity=min(127, c_velo), pitch=int(val), start=int(start), end=int(start + c_dur)))
        except: continue

    if notes:
        track.notes = notes
        midi_obj.instruments.append(track)
        midi_obj.dump(output_path)
        print(f"✅ MIDI saved with {len(notes)} notes.")
    else:
        print("⚠️ Warning: No notes were found in the output tokens.")

# ----------------------------------------------------------
# 3. THE GENERATOR CLASS
# ----------------------------------------------------------
class FigaroGenerator:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")
        self.remi_vocab = RemiVocab()
        self.desc_vocab = DescriptionVocab()
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device).eval()

    def _load_model(self, checkpoint_path):
        pl_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        kwargs = pl_ckpt['hyper_parameters']
        for key in ['flavor', 'vae_run', 'cond_type']:
            if key in kwargs: del kwargs[key]
        
        model = Seq2SeqModule(**kwargs)
        state_dict = pl_ckpt['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith('embeddings.position_ids')}
        state_dict = {k: v for k, v in state_dict.items() if 'crossattention.self.distance_embedding' not in k}

        # Handle size mismatches for expert checkpoint
        cd_size = state_dict.get('desc_in.weight', torch.empty(0)).shape[0]
        cv_size = state_dict.get('in_layer.weight', torch.empty(0)).shape[0]
        if cd_size > 0 and model.desc_in.weight.shape[0] != cd_size:
            model.desc_in = nn.Embedding(cd_size, model.desc_in.embedding_dim)
        if cv_size > 0 and model.in_layer.weight.shape[0] != cv_size:
            model.in_layer = nn.Embedding(cv_size, model.in_layer.embedding_dim)
            model.out_layer = nn.Linear(model.out_layer.in_features, cv_size, bias=False)

        model.load_state_dict(state_dict, strict=False)
        return model

    def generate(self, description_tokens):
        # 🔧 Robust ID lookup
        desc_ids = []
        v_obj = self.desc_vocab.vocab
        for t in description_tokens:
            try:
                if hasattr(v_obj, 'stoi'): idx = v_obj.stoi.get(t, 1)
                elif hasattr(v_obj, 'get_stoi'): idx = v_obj.get_stoi().get(t, 1)
                else: idx = v_obj[t] if t in v_obj else 1
                desc_ids.append(idx)
            except: desc_ids.append(1)
        
        desc_tensor = torch.tensor([desc_ids], dtype=torch.long).to(self.device)
        
        # 🔧 ULTRA PRIME: Start a note completely
        # <bos>=2, Bar_2=114, Position_0=1054, Pitch_60=883, Velocity_20=1334
        # We start the sequence so the AI MUST provide the 'Duration' next.
        prime_ids = [2, 114, 1054, 883, 1334] 
        
        batch = {
            'description': desc_tensor,
            'desc_bar_ids': torch.zeros_like(desc_tensor),
            'input_ids': torch.tensor([prime_ids], device=self.device), 
            'bar_ids': torch.tensor([[0, 2, 2, 2, 2]], device=self.device), # Length 5
            'position_ids': torch.zeros((1, 5), dtype=torch.long).to(self.device),
            'latents': None
        }

        print("🎹 AI is thinking (sampling)...")
        with torch.no_grad():
            try:
                # FIGARO Expert usually expects 'max_length' and 'verbose'
                # We avoid 'do_sample' and 'temperature' which caused crashes.
                raw_output = self.model.sample(
                    batch, 
                    max_length=512,
                    verbose=0
                )
                
                # Extremely defensive unpacking to avoid NoneType errors
                if raw_output is None:
                    print("❌ Model sample() returned None.")
                    return []
                
                if isinstance(raw_output, dict):
                    ids_tensor = raw_output.get('samples', raw_output.get('sequences'))
                    if ids_tensor is not None:
                        ids = ids_tensor[0].cpu().tolist()
                    else:
                        print("❌ Dictionary returned but no 'samples' key found.")
                        return []
                else:
                    ids = raw_output # Assume list

                return self.remi_vocab.decode(ids)
            except Exception as e:
                print(f"❌ Error during sampling logic: {e}")
                return []

# ----------------------------------------------------------
# 4. MAIN EXECUTION
# ----------------------------------------------------------
if __name__ == "__main__":
    timeline = [{'music_instruction': {'chord_sequence': 'C:maj'}, 'midi_constraints': {'program_number': 0}}]
    
    if os.path.exists(CHECKPOINT_PATH):
        try:
            gen = FigaroGenerator(CHECKPOINT_PATH)
            for i, event in enumerate(timeline):
                print(f"\n--- Segment {i} ---")
                # Strong prompt to push AI into music mode
                prompt = [
                    "Bar_2", "Time Signature_4/4", 
                    "Instrument_Acoustic Grand Piano", 
                    "Chord_C:maj", "Note Density_20", 
                    "Mean Velocity_20", "Mean Pitch_60"
                ]
                ai_output = gen.generate(prompt)
                
                print(f"\n📡 FULL AI OUTPUT ({len(ai_output)} tokens):")
                print(ai_output)
                
                if ai_output:
                    robust_save_midi(ai_output, f"segment_{i}.mid")
                else:
                    print("❌ No tokens were generated for this segment.")
        except Exception as e:
            traceback.print_exc()