import os
from PIL import Image

def generate_favicon():
    # Create assets directory if it doesn't exist
    assets_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    # Load the logo
    logo_path = os.path.join(assets_dir, 'logo.png')
    if not os.path.exists(logo_path):
        raise FileNotFoundError(f"Logo file not found at {logo_path}")
        
    # Open and resize logo
    with Image.open(logo_path) as img:
        # Convert to RGBA if not already
        img = img.convert('RGBA')
        
        # Create favicon sizes
        sizes = [(16,16), (32,32), (48,48), (64,64)]
        favicon = img.resize(sizes[-1], Image.Resampling.LANCZOS)
        
        # Save as ICO with multiple sizes
        favicon_path = os.path.join(assets_dir, 'favicon.ico')
        favicon.save(favicon_path, format='ICO', sizes=sizes)
        
        # Also save a PNG version
        favicon_png_path = os.path.join(assets_dir, 'favicon.png')
        favicon.resize((32,32), Image.Resampling.LANCZOS).save(favicon_png_path, format='PNG')
        
        print(f"Generated favicon at {favicon_path}")
        print(f"Generated PNG favicon at {favicon_png_path}")

if __name__ == '__main__':
    generate_favicon()
