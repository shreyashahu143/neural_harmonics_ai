import os
import sys
import torch
import torch.nn as nn
import numpy as np
import miditoolkit
from miditoolkit.midi.containers import Note, Instrument
import traceback
import time

# ----------------------------------------------------------
# 1. SETUP PATHS AND PYTORCH OPTIMIZATION
# ----------------------------------------------------------
torch.set_num_threads(os.cpu_count())

src_path = os.path.abspath(os.path.join(os.getcwd(), 'Rule_Engine', 'figaro', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from Rule_ENgine.figaro.src.models.seq2seq import Seq2SeqModule
from Rule_ENgine.figaro.src.vocab import RemiVocab, DescriptionVocab

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
        # ---- FIX VOCAB CACHING ----
        # Try to find .stoi dict for fast lookup
        if hasattr(self.desc_vocab.vocab, 'get_stoi') and callable(self.desc_vocab.vocab.get_stoi):
            self.v_map = self.desc_vocab.vocab.get_stoi()
        elif hasattr(self.desc_vocab.vocab, 'stoi'):
            self.v_map = self.desc_vocab.vocab.stoi
        else:
            # fallback to .vocab (hopefully a dict {str: int}), or else error
            self.v_map = dict(self.desc_vocab.vocab)
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

        cd_size = state_dict.get('desc_in.weight', torch.empty(0)).shape[0]
        cv_size = state_dict.get('in_layer.weight', torch.empty(0)).shape[0]
        if cd_size > 0 and model.desc_in.weight.shape[0] != cd_size:
            model.desc_in = nn.Embedding(cd_size, model.desc_in.embedding_dim)
        if cv_size > 0 and model.in_layer.weight.shape[0] != cv_size:
            model.in_layer = nn.Embedding(cv_size, model.in_layer.embedding_dim)
            model.out_layer = nn.Linear(model.out_layer.in_features, cv_size, bias=False)

        model.load_state_dict(state_dict, strict=False)
        return model

    def top_p_sampling(self, logits, p=0.94):
        # Standard top-p (nucleus) sampling given logits
        logits = logits.detach().cpu().numpy()
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)

        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        cutoff = np.searchsorted(cumulative_probs, p)
        candidate_indices = sorted_indices[:cutoff+1]

        candidate_probs = probs[candidate_indices]
        candidate_probs /= candidate_probs.sum()

        chosen = np.random.choice(candidate_indices, p=candidate_probs)
        # We must ALWAYS return a cpu-based torch.int64 scalar;
        # we'll place it onto device later.
        return torch.tensor(chosen, dtype=torch.long)

    def generate(self, description_tokens, max_len=64):
        # Step 1: prepare description tensor, using vocab cache for fast lookups as per vocab interface
        get_id = self.v_map.get if hasattr(self.v_map, "get") else lambda x, d=1: self.v_map[x] if x in self.v_map else d
        desc_ids = [get_id(t, 1) for t in description_tokens]
        desc_tensor = torch.tensor([desc_ids], device=self.device, dtype=torch.long)
        desc_bar_ids = torch.zeros_like(desc_tensor) # Required by FIGARO

        generated_ids = [2]  # <bos> is index 2
        print("🎹 Step-by-step Generation (anti-hallucination & live feedback enabled)...")

        for i in range(max_len):
            start_time = time.time()
            # CPU optimization: recreate sequence tensors (1, L) every step
            curr_input = torch.tensor([generated_ids], device=self.device, dtype=torch.long)
            seq_len = curr_input.shape[1]
            curr_bar_ids = torch.zeros((1, seq_len), dtype=torch.long, device=self.device)
            curr_pos_ids = torch.zeros((1, seq_len), dtype=torch.long, device=self.device)

            with torch.no_grad():
                # --- CALL MODEL POSITIONALLY TO AVOID TYPEERROR ---
                # Pass arguments in the required order: 
                # description tensor, current input IDs, description bar IDs, current bar IDs, current position IDs
                output = self.model(desc_tensor, curr_input, desc_bar_ids, curr_bar_ids, curr_pos_ids)
                # Extract logits for last timestep
                if isinstance(output, tuple):
                    logits = output[0][0, -1, :]
                else:
                    logits = output[0, -1, :]  # (vocab_size,)

                # --- ANTI-HALLUCINATION: Strong penalty, especially for Chords ("Chord_" prefix) ---
                # Penalize all tokens in last 20, subtract 10.
                recent_ids = set(generated_ids[-20:])
                for prev_id in recent_ids:
                    logits[prev_id] -= 10.0
                # Further, try to penalize all tokens whose string is a chord token  
                if hasattr(self.remi_vocab, "itos"):
                    vocab_itos = self.remi_vocab.itos
                    for idx in range(len(vocab_itos)):
                        token_str = vocab_itos[idx]
                        if token_str.startswith("Chord_"):
                            logits[idx] -= 5.0 # small uniform penalty to Chord_ tokens

                # --- TOP-P SAMPLING ---
                next_id = self.top_p_sampling(logits, p=0.94).item()
            
            generated_ids.append(next_id)
            # Print live feedback: token #, decoded token, duration for step
            try:
                token_str = self.remi_vocab.itos[next_id]
            except Exception:
                token_str = str(next_id)
            step_time = time.time() - start_time
            print(f"Step {i+1}: generated token '{token_str}' in {step_time:.2f}s", flush=True)

            # STOP: if <pad> (0) or <eos> (3) is predicted, break loop
            if next_id == 0 or next_id == 3:
                print(f"⏹️  Stop token ({token_str}) generated at position {i+1}.")
                break

        return generated_ids

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
                prompt = [
                    "Bar_2", "Time Signature_4/4", 
                    "Instrument_Acoustic Grand Piano", 
                    "Chord_C:maj", "Note Density_20", 
                    "Mean Velocity_25", "Mean Pitch_60", "Mean Duration_8"
                ]
                print(f"💬 Prompt: {prompt}")
                
                ai_output = gen.generate(prompt)  # Default max_len now 64
                
                print(f"\n📡 FULL AI OUTPUT ({len(ai_output)} tokens):")
                print(ai_output)
                
                if ai_output:
                    robust_save_midi(ai_output, f"segment_{i}.mid")
        except Exception as e:
            traceback.print_exc()