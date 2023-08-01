

python convert_original_stable_diffusion_to_diffusers.py \
    --from_safetensors \
    --checkpoint_path="model_weight/safetensor/$1.safetensors" \
    --dump_path="model_weight/$1" \
    --device='cuda:0'