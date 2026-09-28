# ComfyUI-Seedream-Atlas

ComfyUI node for Seedream 4.X Edit Sequential via Atlas Cloud.
1 reference image + prompt in, X images out.

## Install
1. Unzip into `ComfyUI/custom_nodes/` so you get `ComfyUI/custom_nodes/ComfyUI-Seedream-Atlas/__init__.py`
2. `pip install -r requirements.txt` (usually already installed with ComfyUI)
3. Set your key: `export ATLASCLOUD_API_KEY="your-key"` (or paste it in the node)
4. Restart ComfyUI

Node: **Seedream 4.X Edit Sequential (Atlas)** under `ACG/Seedream`

## Outputs
- `images` IMAGE batch
- `urls` newline separated result URLs
- `count` how many images actually came back

## Tips
- Mention the count in your prompt too ("generate 4 images of...")
- `seed` is not sent to the API, it only forces a re-run
- Use `model_override` if a version ID 404s
