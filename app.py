
import os
import re
from pypdf import PdfReader
from pydub import AudioSegment
from f5_tts.api import F5TTS
import torch
import soundfile as sf

def run_tts_system(pdf_path, ref_audio_path):
    os.makedirs('chunks', exist_ok=True)
    
    # Process PDF
    reader = PdfReader(pdf_path)
    sentences = [s.strip() for s in re.split(r'[.!?]', " ".join([p.extract_text() for p in reader.pages])) if s.strip()]
    
    # Trim Audio (5s)
    audio = AudioSegment.from_file(ref_audio_path)[:5000]
    audio.export("trimmed_ref.wav", format="wav")
    
    # Initialize AI
    f5tts = F5TTS(device="cuda" if torch.cuda.is_available() else "cpu")
    
    chunk_paths = []
    for i, text in enumerate(sentences):
        out = f"chunks/chunk_{i:04d}.wav"
        wav, sr, _ = f5tts.infer(gen_text=text, ref_file="trimmed_ref.wav", ref_text="")
        sf.write(out, wav, sr)
        chunk_paths.append(out)
    
    # Merge
    combined = AudioSegment.empty()
    for p in sorted(chunk_paths):
        combined += AudioSegment.from_wav(p) + AudioSegment.silent(duration=300)
    combined.export("output_full.wav", format="wav")
    print("Done!")

if __name__ == '__main__':
    # You can customize these paths
    run_tts_system('input.pdf', 'reference.mp3')
