# AMD Installation

The required packages are in the file `requirements.txt`, and you can run the following command to install the environment:

```
conda create --name amd python=3.8 -y
conda activate amd

conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch

pip install -r requirements.txt
```

### Note:
- **The above commands are for reference only**, please configure your own environment according to your needs.
- We recommend installing **`PyTorch >= 1.12.0`**, which may greatly reduce the GPU memory usage.
- If you encounter issues when installing `PyTorch` using `conda`, you can try installing it via wheels. You can find suitable versions of the package by searching on [website](https://download.pytorch.org/whl/torch_stable.html).
- It is recommended to install **`timm == 0.4.12`**, because some of the APIs we use are deprecated in the latest version of timm.
- If you encounter any problems while installing deepspeed, you can refer [here](https://www.deepspeed.ai/tutorials/advanced-install/). You can also choose not to install deepspeed, just remove `--enable_deepspeed` from the script.