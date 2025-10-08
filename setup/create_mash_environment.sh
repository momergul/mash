#!/bash/bin

# Drawn from verl
# Install verl
cd verl
#pip install -e .

# Install vllm
#pip install vllm==0.8.2

# Install flash-attn
#pip install flash-attn=2.7.4.post1 --no-build-isolation

cd ..
mv verl verl_top
mv verl_top/verl verl
