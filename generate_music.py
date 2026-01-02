import os
import sys
import torch
import numpy as np
from test import process_emotion_sequence

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
        # --- 1. PREPARE DESCRIPTION (ENCODER) ---
        desc_ids = []

        # Safe Encode Description
        for token in description:
            try:
                if hasattr(self.desc_vocab.vocab, '__getitem__'):
                    desc_ids.append(self.desc_vocab.vocab[token])
                elif hasattr(self.desc_vocab.vocab, 'lookup_indices'):
                    desc_ids.append(self.desc_vocab.vocab.lookup_indices([token])[0])
                else:
                    desc_ids.append(self.desc_vocab.vocab([token])[0])
            except:
                desc_ids.append(0)

        # Add BOS to description if needed (some models expect it, some don't, safety first)
        # (Keeping your previous logic of just tokens is fine, let's stick to what we had)
        
        # Create Description Tensor
        desc_tensor = torch.tensor(desc_ids, dtype=torch.long)

        # Calculate Description Bar IDs (For the Encoder)
        desc_bar_ids = torch.zeros(len(desc_ids), dtype=torch.long)
        current_bar = 0
        for i, token_id in enumerate(desc_ids):
            token_str = ""
            try:
                if isinstance(token_id, torch.Tensor): token_id = token_id.item()
                if hasattr(self.desc_vocab.vocab, 'lookup_token'):
                    token_str = self.desc_vocab.vocab.lookup_token(token_id)
                elif hasattr(self.desc_vocab.vocab, 'itos'):
                    token_str = self.desc_vocab.vocab.itos[token_id]
            except: pass
            if "Bar_" in token_str:
                try: current_bar = int(token_str.split('_')[1])
                except: pass
            desc_bar_ids[i] = current_bar

        # --- 2. PREPARE INPUT (DECODER START) ---
        # The decoder starts with just ONE token: The BOS (Start) token.
        try:
            bos_id = self.desc_vocab.vocab[BOS_TOKEN] if hasattr(self.desc_vocab.vocab, '__getitem__') else 1
        except:
            bos_id = 1

        # Input ID: Shape [1, 1]
        input_ids = torch.tensor([[bos_id]], dtype=torch.long).to(self.device)

        # Bar ID for Input: Shape [1, 1] (It starts at Bar 0)
        input_bar_ids = torch.zeros((1, 1), dtype=torch.long).to(self.device)

        # Position ID for Input: Shape [1, 1] (It starts at Position 0)
        input_position_ids = torch.zeros((1, 1), dtype=torch.long).to(self.device)

        # --- 3. RETURN BATCH ---
        return {
            # Encoder Inputs (Long)
            'description': desc_tensor.unsqueeze(0).to(self.device),
            'desc_bar_ids': desc_bar_ids.unsqueeze(0).to(self.device),
            
            # Decoder Inputs (Short - just the start token)
            'input_ids': input_ids,
            'bar_ids': input_bar_ids,          # Matches input_ids shape
            'position_ids': input_position_ids,# Matches input_ids shape
            
            'latents': None
        }

    def sample(self, description, **kwargs):
        batch = self._prepare_batch(description)
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            # The inner model doesn't accept 'temperature', so we remove it if present
            kwargs.pop('temperature', None)
            return self.model.sample(batch, verbose=0, **kwargs)

    def save_midi(self, result, output_path):
        seq = result['sequences'][0].cpu().numpy()
        events = self.remi_vocab.decode(seq)
        pm = remi2midi(events)
        pm.write(output_path)

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

    for i, event in enumerate(timeline_data):
        # Calculate how many bars for this duration (Assuming 2s per bar approx)
        duration_sec = event['end_time_sec'] - event['start_time_sec']
        num_bars = max(1, int(duration_sec / 2.0))
        
        # Create the Description for *one bar*
        # (We use bar_num=1 for all, treating each segment as a fresh start for now)
        bar_tokens = translate_rules_to_figaro_tokens(event, bar_num=1)
        
        # Repeat this bar description for the duration
        # Figaro wants a List of Lists (one list per bar)
        full_description = [bar_tokens for _ in range(num_bars)]
        
        print(f"\n--- Segment {i+1}: Generating {num_bars} bars ---")
        print(f"Tokens: {bar_tokens}")

        # GENERATE
        # Note: We pass the sequence of bar descriptions
        result = model.sample(
            description=full_description, 
            temperature=1.0
        )
        
        # SAVE
        # Create a clean filename
        safe_chord = event['music_instruction']['chord_sequence'].replace(':', '')
        filename = f"output_seg_{i}_{safe_chord}.mid"
        model.save_midi(result, filename)
        print(f"✅ Saved: {filename}")

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
        generate_song_from_rules(timeline)
    else:
        print(f"❌ Checkpoint missing at {CHECKPOINT_PATH}")