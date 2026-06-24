This Python script analyzes an image containing text, lets the user mask a selected region, and then attempts to reconstruct the missing word using glyphs already present elsewhere in the image

The pipeline combines :

image preprocessing (OpenCV),    
line segmentation with Kraken,         
glyph extraction and deduplication,           
character and word spacing analysis,        
candidate sequence generation,             
OCR with Tesseract,          
dictionary filtering,               
and LLM-based ranking via Ollama

Dependencies

The script uses the following :

Python packages :          
opencv-python,    
numpy,     
scikit-learn

External tools :      
Kraken,   
Tesseract OCR,   
Ollama 
