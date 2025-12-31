# main_sequencer.py (New Concept)
from Rule_ENgine.src.rules_engine import MusicRulesEngine
import json

# 1. THE INPUT (Received from NLP Team)
# A list of vectors representing the flow of the text
nlp_batch_output = [
    # "I was really excited..." (0s - 5s)
    {"valence": 0.8, "arousal": 0.7, "dominance": 0.6},
    
    # "...but then everything fell apart." (5s - 10s)
    {"valence": -0.8, "arousal": 0.6, "dominance": 0.2},
    
    # "Now I don't know what to do." (10s - 15s)
    {"valence": -0.4, "arousal": 0.1, "dominance": 0.1}
]

# User Constraints
user_config = {
    "instrument": "piano", 
    "total_duration": 15 # Seconds for the whole clip
}

# 2. THE PROCESSOR
def process_emotion_sequence(nlp_list, config):
    engine = MusicRulesEngine()
    timeline = []
    
    # Calculate duration per segment (Simple Division)
    # Or NLP could provide specific timestamps
    segment_duration = config['total_duration'] / len(nlp_list)
    
    print(f"--- Processing {len(nlp_list)} Emotional Segments ---")
    
    for i, vector in enumerate(nlp_list):
        # Update the duration for this specific chunk
        current_config = {
            "instrument": config['instrument'],
            "duration": segment_duration
        }
        
        # RUN THE ENGINE
        # The engine remembers the 'prev_vector' internally automatically!
        result = engine.process(vector, current_config)
        
        # Add timestamp info for the Transformer
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        
        timeline_event = {
            "start_time_sec": round(start_time, 1),
            "end_time_sec": round(end_time, 1),
            "music_instruction": result['figaro_prompt'], # The "Prompt" part
            "midi_constraints": result['midi_constraints']
        }
        
        timeline.append(timeline_event)

    return timeline

# 3. EXECUTE
final_sequence = process_emotion_sequence(nlp_batch_output, user_config)

# 4. OUTPUT TO TRANSFORMER
print(json.dumps(final_sequence, indent=2))