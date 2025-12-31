import numpy as np
import math

# --- THE FIX: HARDCODED ANCHOR ---
# We always start at Middle C. The Transformer handles the rest.
ANCHOR_NOTE_MIDI = 60  # Middle C
ANCHOR_NOTE_NAME = "C"

class MusicRulesEngine:
    def __init__(self):
        self.prev_vector = None
        self.prev_quadrant = None

    # ... (Keep your existing _get_midi_program and _sigmoid methods) ...
    def _get_midi_program(self, instrument_name, conflict_score):
        # [Same instrument mapping logic as before]
        name = instrument_name.lower()
        mapping = {
            "piano": 0, "acoustic guitar": 24, "electric guitar": 27,
            "harp": 46, "flute": 73, "violin": 40, "cello": 42
        }
        prog_id = mapping.get(name, 0)
        if name == "electric guitar" and conflict_score > 0.6:
            prog_id = 30 # Distortion
        return prog_id

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _get_quadrant(self, v, a):
        if v >= 0 and a >= 0.5: return 1
        if v < 0 and a >= 0.5: return 2
        if v < 0 and a < 0.5: return 3
        return 4

    def _get_chord_symbol(self, v, a):
        """
        Returns the CHORD SYMBOL relative to C.
        This is what Figaro/Transformer needs to know what to play.
        """
        # 1. High Energy + Positive = Heroic (Mixolydian)
        if v > 0.3 and a > 0.6:
            quality = "mix" # C Mixolydian
            mode_full = "Mixolydian"
            
        # 2. Very Positive = Wonder (Lydian)
        elif v > 0.6: 
            quality = "lyd" # C Lydian
            mode_full = "Lydian"
            
        # 3. Positive = Happy (Major)
        elif v > 0.2: 
            quality = "maj" # C Major
            mode_full = "Major"
            
        # 4. Slightly Negative = Folk/Ancient (Dorian)
        elif v > -0.2: 
            quality = "dor" # C Dorian
            mode_full = "Dorian"
            
        # 5. Negative = Sad (Minor)
        elif v > -0.6: 
            quality = "min" # C Minor
            mode_full = "Minor"
            
        # 6. Very Negative = Evil/Horror
        else:
            if a > 0.5:
                quality = "loc" # C Locrian (Unstable/Evil)
                mode_full = "Locrian"
            else:
                quality = "phr" # C Phrygian (Dark/Tension)
                mode_full = "Phrygian"

        # Override: Panic (Diminished)
        if a > 0.9 and v < -0.8:
            quality = "dim"
            mode_full = "Diminished"

        # THE OUTPUT: "C:min", "C:maj", "C:dim"
        return f"{ANCHOR_NOTE_NAME}:{quality}", mode_full

    def process(self, nlp_data, user_config):
        v = nlp_data.get('valence', 0.0)
        a = nlp_data.get('arousal', 0.0)
        d = nlp_data.get('dominance', 0.0)
        
        user_inst = user_config.get('instrument', 'piano')
        user_dur = user_config.get('duration', 10)

        # 1. Logic Calcs
        tension = (a * 0.5) + ((1.0 - v) * 0.3)
        conflict = a * (1.0 - self._sigmoid(v * 5))
        
        # 2. Get Symbolic Data (The Fix)
        chord_symbol, mode_name = self._get_chord_symbol(v, a)
        
        # 3. Instrument Setup
        midi_prog = self._get_midi_program(user_inst, conflict)
        
        # 4. Octave Logic (Still useful for Instrument constraints)
        # Even if Key is C, Cello plays C2 (36), Flute plays C5 (72)
        base_octave = 0
        if "cello" in user_inst or "bass" in user_inst: base_octave = -2
        if "flute" in user_inst or "violin" in user_inst: base_octave = +1
        
        # Arousal shift (Higher energy = Higher pitch)
        energy_shift = 1 if a > 0.7 else 0
        
        final_root_midi = ANCHOR_NOTE_MIDI + (base_octave * 12) + (energy_shift * 12)

        # Motion Logic (kept simple)
        current_quadrant = self._get_quadrant(v, a)
        motif_action = "new_theme" if current_quadrant != self.prev_quadrant else "continue"
        self.prev_quadrant = current_quadrant

        return {
            "meta": {
                "duration_sec": user_dur,
                "bpm": int(60 + (a * 100))
            },
            "figaro_prompt": {
                # This is the string you feed into the Transformer
                "chord_sequence": f"{chord_symbol}", 
                "key": ANCHOR_NOTE_NAME,
                "mode": mode_name,
                "tempo": "Fast" if a > 0.6 else "Slow",
                "density": "High" if a > 0.5 else "Low"
            },
            "midi_constraints": {
                "program_number": midi_prog,
                "root_note_midi": final_root_midi, # C3, C4, or C5 depending on instrument
                "scale_lock": mode_name
            }
        }