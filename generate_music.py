import os
import sys
import torch
import numpy as np
from test import process_emotion_sequence
import miditoolkit
from miditoolkit.midi.containers import Note, Instrument, Marker, TimeSignature
import re

# ----------------------------------------------------------
# 1. SETUP PATHS - Give access to internal Figaro code

# Make src_path absolute, robust, and high-priority in sys.path
src_path = os.path.abspath(os.path.join(os.getcwd(), 'Rule_Engine', 'figaro', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# This ensures that imports like "import vocab" in src code work for sibling modules
# (This path must be in sys.path before any figaro imports).

# ----------------------------------------------------------

# Try to import pretty_midi for instrument names (Standard in Figaro env)
try:
    import pretty_midi
except ImportError:
    print("⚠️ pretty_midi not found. Using manual instrument lookup.")
    pretty_midi = None

# ----------------------------------------------------------
# 2. INTERNAL IMPORTS for the wrapper (CRASH on missing modules! No try/except)
from Rule_ENgine.figaro.src.models.seq2seq import Seq2SeqModule
from Rule_ENgine.figaro.src.vocab import RemiVocab, DescriptionVocab
from Rule_ENgine.figaro.src.constants import BOS_TOKEN, EOS_TOKEN, BAR_KEY
from Rule_ENgine.figaro.src.input_representation import remi2midi
from transformers.models.bert.modeling_bert import BertAttention
# ----------------------------------------------------------

CHECKPOINT_PATH = os.path.join("Rule_ENgine", "figaro", "checkpoints", "figaro-expert.ckpt")

# 3. Define FigaroGenerator as a Wrapper Class
class FigaroGenerator:
    def __init__(self, checkpoint_path, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Loading model from {checkpoint_path}...")
        self.desc_vocab = DescriptionVocab()
        self.remi_vocab = RemiVocab()
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.model.freeze()

    def _load_model(self, checkpoint_path):
        import torch.nn as nn
        
        pl_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        kwargs = pl_ckpt['hyper_parameters']
        for key in ['flavor', 'vae_run']:
            if key in kwargs: del kwargs[key]
        model = Seq2SeqModule(**kwargs)
        
        # Get state dict and filter out position_ids
        state_dict = pl_ckpt['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith('embeddings.position_ids')}
        
        # Filter out unexpected crossattention distance_embedding keys
        state_dict = {k: v for k, v in state_dict.items() 
                     if 'crossattention.self.distance_embedding' not in k}
        
        # Patch: Resize layers to match checkpoint sizes (Checkpoint has +3 tokens)
        # Check checkpoint sizes from state_dict
        checkpoint_desc_size = state_dict.get('desc_in.weight', torch.empty(0)).shape[0]
        checkpoint_vocab_size = state_dict.get('in_layer.weight', torch.empty(0)).shape[0]
        
        # Resize Description Input
        if checkpoint_desc_size > 0 and hasattr(model, 'desc_in') and model.desc_in.weight.shape[0] != checkpoint_desc_size:
            print(f"⚠️ Resizing desc_in from {model.desc_in.weight.shape[0]} to {checkpoint_desc_size}")
            embedding_dim = model.desc_in.embedding_dim
            new_desc = nn.Embedding(checkpoint_desc_size, embedding_dim)
            model.desc_in = new_desc

        # Resize Input Layer
        if checkpoint_vocab_size > 0 and model.in_layer.weight.shape[0] != checkpoint_vocab_size:
            print(f"⚠️ Resizing in_layer from {model.in_layer.weight.shape[0]} to {checkpoint_vocab_size}")
            embedding_dim = model.in_layer.embedding_dim
            new_in = nn.Embedding(checkpoint_vocab_size, embedding_dim)
            model.in_layer = new_in

        # Resize Output Layer
        if checkpoint_vocab_size > 0 and model.out_layer.weight.shape[0] != checkpoint_vocab_size:
            print(f"⚠️ Resizing out_layer from {model.out_layer.weight.shape[0]} to {checkpoint_vocab_size}")
            in_features = model.out_layer.in_features
            new_out = nn.Linear(in_features, checkpoint_vocab_size, bias=False)
            model.out_layer = new_out
        
        # Try loading state dict with error handling
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            # Handle crossattention configuration issues
            config = model.transformer.decoder.bert.config
            for layer in model.transformer.decoder.bert.encoder.layer:
                layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
            model.load_state_dict(state_dict, strict=False)
        
        return model

    def _prepare_batch(self, description):
        # 1. Prepare Description
        desc_ids = []
        for token in description:
            try:
                if hasattr(self.desc_vocab.vocab, 'lookup_indices'):
                    desc_ids.append(self.desc_vocab.vocab.lookup_indices([token])[0])
                else:
                    desc_ids.append(self.desc_vocab.vocab[token])
            except:
                desc_ids.append(0)
        desc_tensor = torch.tensor(desc_ids, dtype=torch.long)

        # 2. PERFECT PRIME (Full Note Definition)
        # Sequence: <bos> | Bar_2 | Position_0 | Pitch_60 | Velocity_20 | Duration_4
        # IDs from your dump: 
        # <bos>=2, Bar_2=114, Position_0=1054, Pitch_60=883, Velocity_20=1334, Duration_4=645

        prime_sequence = [2, 114, 1054, 883, 1334, 645]

        input_ids = torch.tensor([prime_sequence], dtype=torch.long).to(self.device)

        # Bar IDs: [0, 2, 2, 2, 2, 2]
        input_bar_ids = torch.tensor([[0, 2, 2, 2, 2, 2]], dtype=torch.long).to(self.device)

        # Position IDs: [0, 0, 0, 0, 0, 0]
        input_position_ids = torch.zeros((1, 6), dtype=torch.long).to(self.device)

        print(f"🚀 PERFECT PRIME: Forcing complete note with {len(prime_sequence)} tokens...")

        return {
            'description': desc_tensor.unsqueeze(0).to(self.device),
            'desc_bar_ids': torch.zeros_like(desc_tensor).unsqueeze(0).to(self.device),
            'input_ids': input_ids,
            'bar_ids': input_bar_ids,
            'position_ids': input_position_ids,
            'latents': None
        }

    def sample(self, description, **kwargs):
        batch = self._prepare_batch(description)
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            # The inner model doesn't accept 'temperature', so we remove it if present
            kwargs.pop('temperature', None)
            return self.model.sample(batch, verbose=0, **kwargs)


    def save_midi(tokens, output_path):
        import miditoolkit
        from miditoolkit.midi.containers import Note, Instrument
        
        print(f"\n🔍 DEBUG START: Processing {len(tokens)} tokens for {output_path}")

        # Configuration based on your vocab_dump.txt
        TOKEN_MAP = {
            'PITCH':    ['Pitch', 'Note_On'],
            'VELOCITY': ['Velocity', 'Vel'],
            'DURATION': ['Duration', 'Dur'],
            'BAR':      ['Bar'],
            'POSITION': ['Position']
        }

        midi_obj = miditoolkit.midi.parser.MidiFile()
        track = Instrument(program=0, is_drum=False, name="Debug Track")
        
        state = {'bar': 0, 'position': 0, 'pitch': 0, 'velocity': 64, 'duration': 4}
        TICKS_PER_BAR, TICKS_PER_POS = 1920, 15
        notes_to_add = []

        for i, token in enumerate(tokens):
            # 1. Skip meta tokens
            if "<" in token:
                print(f"  [{i}] Skipping meta-token: {token}")
                continue
                
            # 2. Split and Validate
            parts = token.split('_')
            if len(parts) < 2:
                print(f"  [{i}] Skipping non-data token: {token}")
                continue
                
            prefix, value_str = parts[0], parts[-1]
            try:
                value = int(value_str)
            except:
                print(f"  [{i}] Error: Could not convert '{value_str}' to number in {token}")
                continue

            # 3. Match Categories
            matched = False
            
            if any(alias in prefix for alias in TOKEN_MAP['PITCH']):
                state['pitch'] = value
                start = (state['bar'] * TICKS_PER_BAR) + (state['position'] * TICKS_PER_POS)
                end = start + (state['duration'] * TICKS_PER_POS)
                
                new_note = Note(velocity=state['velocity'], pitch=state['pitch'], start=int(start), end=int(end))
                notes_to_add.append(new_note)
                print(f"  [{i}] ✨ ADDED NOTE: Pitch {value} at Bar {state['bar']}, Pos {state['position']}")
                matched = True

            elif any(alias in prefix for alias in TOKEN_MAP['BAR']):
                state['bar'] = value
                print(f"  [{i}] 🆕 BAR SET TO: {value}")
                matched = True

            elif any(alias in prefix for alias in TOKEN_MAP['POSITION']):
                state['position'] = value
                print(f"  [{i}] 📍 POSITION SET TO: {value}")
                matched = True

            elif any(alias in prefix for alias in TOKEN_MAP['VELOCITY']):
                state['velocity'] = value
                print(f"  [{i}] 🔊 VELOCITY SET TO: {value}")
                matched = True

            elif any(alias in prefix for alias in TOKEN_MAP['DURATION']):
                state['duration'] = value
                print(f"  [{i}] ⏳ DURATION SET TO: {value}")
                matched = True

            if not matched:
                print(f"  [{i}] ⚠️ IGNORED: Token '{token}' did not match any category.")

        # 4. Final Save Report
        print(f"📊 DEBUG SUMMARY: Found {len(notes_to_add)} valid notes to write.")
        
        if len(notes_to_add) > 0:
            track.notes = notes_to_add
            midi_obj.instruments.append(track)
            midi_obj.dump(output_path)
            print(f"✅ SUCCESS: File saved with content.")
        else:
            print(f"❌ FAILURE: No notes were added. The file will be empty (41 bytes).")

# --- 2. HELPER: BINNING LOGIC ---
# We mimic Figaro's internal binning to convert raw values (0-127) to indices (0-31)
def get_bin_index(value, bins):
    """Finds the closest bin index for a given value."""
    return np.argmin(np.abs(bins - value))

# Define the bins exactly as Figaro does (from your notes)
DENSITY_BINS = np.linspace(0, 12, 33)      # 0 to 12 notes/pos
VELOCITY_BINS = np.linspace(0, 128, 33)    # 0 to 127 volume
PITCH_BINS = np.linspace(0, 128, 33)       # 0 to 127 pitch
DURATION_BINS = np.logspace(0, 7, 33, base=2) # Logarithmic duration

def get_instrument_name(program_id):
    """Converts MIDI ID to the specific string Figaro expects."""
    if pretty_midi:
        return pretty_midi.program_to_instrument_name(program_id)
    
    # Manual fallback for your 7 instruments if library fails
    lookup = {
        0: "Acoustic Grand Piano",
        24: "Acoustic Guitar (nylon)",
        27: "Electric Guitar (clean)",
        30: "Overdriven Guitar",
        40: "Violin",
        42: "Cello",
        46: "Orchestral Harp",
        73: "Flute"
    }
    return lookup.get(program_id, "Acoustic Grand Piano")

# --- 3. THE TRANSLATOR (Rules -> Tokens) ---
def translate_rules_to_figaro_tokens(event, bar_num=1):
    """
    Converts One Segment -> A List of Token Strings for ONE BAR.
    """
    instr = event['music_instruction']
    midi_rules = event['midi_constraints']
    
    # A. Calculate Values from "High/Low" strings
    # 1. Density
    density_map = {"Low": 2.0, "Medium": 6.0, "High": 10.0} # Notes per position
    raw_density = density_map.get(instr.get('density', "Medium"), 6.0)
    density_idx = get_bin_index(raw_density, DENSITY_BINS)
    
    # 2. Velocity (Volume)
    # 120=Loud, 80=Medium, 50=Soft
    velocity_idx = get_bin_index(80, VELOCITY_BINS) 

    # 3. Mean Pitch (Register)
    # We use the 'root_note' as the center pitch
    pitch_idx = get_bin_index(midi_rules['root_note_midi'], PITCH_BINS)

    # 4. Mean Duration (Tempo/Speed approximation)
    # Fast tempo = Shorter notes (lower duration bin)
    dur_val = 2.0 if instr['tempo'] == "Fast" else 8.0
    duration_idx = get_bin_index(dur_val, DURATION_BINS)

    # B. Get Strings
    inst_name = get_instrument_name(midi_rules['program_number'])
    chord_str = f"Chord_{instr['chord_sequence']}" # e.g. "Chord_C:min"

    # C. Construct the Token List
    # Strict Order: Bar -> TimeSig -> Meta -> Inst -> Chord
    tokens = [
        f"Bar_{bar_num}",
        "Time Signature_4/4",
        f"Note Density_{density_idx}",
        f"Mean Velocity_{velocity_idx}",
        f"Mean Pitch_{pitch_idx}",
        f"Mean Duration_{duration_idx}",
        f"Instrument_{inst_name}",
        chord_str
    ]
    
    return tokens

# --- 4. MAIN GENERATION LOOP ---
def generate_song_from_rules(timeline_data):
    print(f"🔌 Loading Expert Model from: {CHECKPOINT_PATH}")
    model = FigaroGenerator(
        checkpoint_path=CHECKPOINT_PATH,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("🎹 Model Loaded! Starting Composition...")

    # --- DEBUG: INSPECT VOCABULARY ---
    print("\n🔍 INSPECTING MUSIC VOCABULARY (REMI)...")
    try:
        vocab_obj = model.remi_vocab.vocab
        if hasattr(vocab_obj, 'itos'):
            print(f"First 10 tokens: {vocab_obj.itos[:10]}")
            bar_tokens = [t for t in vocab_obj.itos if "Bar" in str(t)][:5]
            print(f"Found 'Bar' tokens: {bar_tokens}")
        elif hasattr(vocab_obj, 'lookup_token'):
            sample_tokens = [vocab_obj.lookup_token(i) for i in range(10)]
            print(f"First 10 tokens: {sample_tokens}")
        elif hasattr(vocab_obj, 'stoi'):
            print("Using STOI dictionary...")
            sample_items = list(vocab_obj.stoi.items())[:10]
            print(f"First 10 items: {sample_items}")
    except Exception as e:
        print(f"Could not inspect vocab: {e}")
    print("--------------------------------------\n")

    def save_midi(token_list, output_path):
        pm = remi2midi(token_list)
        pm.write(output_path)

    for i, event in enumerate(timeline_data):
        duration_sec = event['end_time_sec'] - event['start_time_sec']
        num_bars = max(1, int(duration_sec / 2.0))
        bar_tokens = translate_rules_to_figaro_tokens(event, bar_num=1)
        full_description = [bar_tokens for _ in range(num_bars)]

        print(f"\n--- Segment {i+1}: Generating {num_bars} bars ---")
        print(f"Tokens: {bar_tokens}")

        print(f"🎹 Generating Segment {i+1}...")

        # --- NEW GENERATION/BLOCK FOR DICT SAFE-UNPACK ---
        try:

            # 1. Generate music
            raw_output = model.sample(
                description=full_description,
                temperature=1.0
            )
            
            # 2. UNPACK: Check if it's a dictionary or a list
            if isinstance(raw_output, dict):
                # The tokens are usually in 'samples' or 'sequences'
                generated_tokens = raw_output.get('samples', raw_output.get('sequences', []))
                # If it's a tensor, convert to list
                if hasattr(generated_tokens, 'tolist'):
                    generated_tokens = generated_tokens.tolist()
                # If it's a list of lists (batch), take the first one
                if len(generated_tokens) > 0 and isinstance(generated_tokens[0], list):
                    generated_tokens = generated_tokens[0]
            else:
                generated_tokens = raw_output

            # 3. ENSURE STRINGS: If they are still IDs, decode them
            # (This handles cases where the model returns numbers instead of words)
            if len(generated_tokens) > 0 and isinstance(generated_tokens[0], int):
                generated_tokens = model.remi_vocab.decode(generated_tokens)

            #print(f"📡 AI Output (first 5): {generated_tokens[:5]}")
            print(f"📡 FULL AI OUTPUT ({len(generated_tokens)} tokens):")
            print(generated_tokens)
            # 4. INJECT: Add Bar_1 manually for the MIDI writer
            if generated_tokens and "Bar_" not in str(generated_tokens[0]):
                print("💉 Injecting 'Bar_1' to start the MIDI track...")
                generated_tokens.insert(0, "Bar_1")

            # 5. SAVE
            output_filename = f"output_seg_{i}.mid"
            save_midi(generated_tokens, output_filename)
            print(f"✅ Success! Saved to {output_filename}")

        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc() # This will show us exactly where it crashes if it fails again

if __name__ == "__main__":
    # 1. Simulate NLP Input
    nlp_input = [
        {"valence": -0.8, "arousal": 0.6, "dominance": 0.2}, # Anguish
        {"valence": 0.5, "arousal": 0.2, "dominance": 0.8}   # Resolution
    ]
    config = {"instrument": "violin", "total_duration": 12}
    
    # 2. Get Rules
    timeline = process_emotion_sequence(nlp_input, config)

    # 3. Generate
    if os.path.exists(CHECKPOINT_PATH):
        # --- TOOL: DUMP VOCABULARY TO FILE ---
        print("\n📝 Dumping full vocabulary to 'vocab_dump.txt'...")
        try:
            model = FigaroGenerator(
                checkpoint_path=CHECKPOINT_PATH,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            vocab_obj = model.remi_vocab.vocab

            all_tokens = []
            if hasattr(vocab_obj, 'itos'):
                all_tokens = vocab_obj.itos
            elif hasattr(vocab_obj, 'get_itos'):
                all_tokens = vocab_obj.get_itos()
            elif hasattr(vocab_obj, 'lookup_token'):
                try:
                    for i in range(5000): # Safety limit
                        all_tokens.append(vocab_obj.lookup_token(i))
                except:
                    pass

            with open("vocab_dump.txt", "w", encoding="utf-8") as f:
                for idx, token in enumerate(all_tokens):
                    f.write(f"{idx}: {token}\n")

            print(f"✅ Success! Saved {len(all_tokens)} tokens to 'vocab_dump.txt'")
            print("👉 Open that file to find the EXACT name for 'Bar_1' (e.g. 'Bar_1', 'Bar_0', 'Subject_Bar_1')")

        except Exception as e:
            print(f"❌ Could not dump vocab: {e}")
        print("--------------------------------------\n")
        generate_song_from_rules(timeline)
    else:
        print(f"❌ Checkpoint missing at {CHECKPOINT_PATH}")