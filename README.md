# Difix3D gsplat Demo

**Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models**  
[Jay Zhangjie Wu*](https://zhangjiewu.github.io/), [Yuxuan Zhang*](https://scholar.google.com/citations?user=Jt5VvNgAAAAJ&hl=en), [Haithem Turki](https://haithemturki.com/), [Xuanchi Ren](https://xuanchiren.com/), [Jun Gao](https://www.cs.toronto.edu/~jungao/),  
[Mike Zheng Shou](https://sites.google.com/view/showlab/home?authuser=0), [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/), [Zan Gojcic†](https://zgojcic.github.io/), [Huan Ling†](https://www.cs.toronto.edu/~linghuan/) _(*/† equal contribution/advising)_  
CVPR 2025 (Oral)  
[Project Page](https://research.nvidia.com/labs/toronto-ai/difix3d/) | [Paper](https://arxiv.org/abs/2503.01774) | [Model](https://huggingface.co/nvidia/difix) | [Demo](https://huggingface.co/spaces/nvidia/difix)

<div align="center">
  <img src="assets/demo.gif" alt=""  width="1100" />
</div>

## Setup using `uv`
make sure you install `uv` first
```bas
git clone git@github.com:ligaoqi2/Difix-splat-demo.git
cd Difix-splat-demo
uv sync
```

## Quickstart (diffusers)

```
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

input_image = load_image("assets/example_input.png")
prompt = "remove degradation"

output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
output_image.save("example_output.png")
```

Optionally, you can use a reference image to guide the denoising process.
```
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
pipe.to("cuda")

input_image = load_image("assets/example_input.png")
ref_image = load_image("assets/example_ref.png")
prompt = "remove degradation"

output_image = pipe(prompt, image=input_image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
output_image.save("example_output.png")
```

## Difix: Single-step diffusion for 3D artifact removal

### Training

#### Data Preparation

Prepare your dataset in the following JSON format:

```json
{
  "train": {
    "{data_id}": {
      "image": "{PATH_TO_IMAGE}",
      "target_image": "{PATH_TO_TARGET_IMAGE}",
      "ref_image": "{PATH_TO_REF_IMAGE}",
      "prompt": "remove degradation"
    }
  },
  "test": {
    "{data_id}": {
      "image": "{PATH_TO_IMAGE}",
      "target_image": "{PATH_TO_TARGET_IMAGE}",
      "ref_image": "{PATH_TO_REF_IMAGE}",
      "prompt": "remove degradation"
    }
  }
}
```

#### Single GPU

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/difix/train \
    --dataset_path="data/data.json" \
    --max_train_steps 10000 \
    --resolution=512 --learning_rate 2e-5 \
    --train_batch_size=1 --dataloader_num_workers 8 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=1000 --eval_freq 1000 --viz_freq 100 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
    --report_to "wandb" --tracker_project_name "difix" --tracker_run_name "train" --timestep 199
```

#### Multipe GPUs

```bash
export NUM_NODES=1
export NUM_GPUS=8
accelerate launch --mixed_precision=bf16 --main_process_port 29501 --multi_gpu --num_machines $NUM_NODES --num_processes $NUM_GPUS src/train_difix.py \
    --output_dir=./outputs/difix/train \
    --dataset_path="data/data.json" \
    --max_train_steps 10000 \
    --resolution=512 --learning_rate 2e-5 \
    --train_batch_size=1 --dataloader_num_workers 8 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=1000 --eval_freq 1000 --viz_freq 100 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
    --report_to "wandb" --tracker_project_name "difix" --tracker_run_name "train" --timestep 199
```

### Inference

Place the `model_*.pkl` in the `checkpoints` directory. You can run inference using the following command:

```bash
python src/inference_difix.py \
    --model_path "checkpoints/model.pkl" \
    --input_image "assets/example_input.png" \
    --prompt "remove degradation" \
    --output_dir "outputs/difix" \
    --timestep 199
```


## Difix3D: Progressive 3D update

### Data Format

The data should be organized in the following structure:

```
DATA_DIR/
├── {SCENE_ID}
│   ├── colmap
│   │   ├── sparse
│   │   │   └── 0
│   │   │       ├── cameras.bin
│   │   │       ├── database.db
│   │   │       └── ...
│   ├── images
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   ├── ...
│   │   ├── 0200.png
│   │   ├── 0201.png
│   │   └── ...
│   ├── images_2
│   ├── images_4
│   └── images_8
```

### gsplat

```bash
uv add git+https://github.com/nerfstudio-project/gsplat.git@v1.5.2 --no-build-isolation
```

Run Difix3D finetuning with gsplat.
```bash
SCENE_ID=032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7
DATA=DATA_DIR/${SCENE_ID}/gaussian_splat
DATA_FACTOR=4
CKPT_PATH=CKPT_DIR/${SCENE_ID}/ckpts/ckpt_29999_rank0.pt # Path to the pretrained checkpoint file
OUTPUT_DIR=outputs/difix3d/gsplat/${SCENE_ID}

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA} --data_factor ${DATA_FACTOR} \
    --result_dir ${OUTPUT_DIR} --no-normalize-world-space --test_every 1 --ckpt ${CKPT_PATH}
```


## Difix3D+: With real-time post-rendering

Due to the limited capacity of reconstruction methods to represent sharp details, some regions remain blurry. To further enhance the novel views, we use our Difix model as the final post-processing step at render time.

```bash
python src/inference_difix.py \
    --model_path "checkpoints/model.pkl" \
    --input_image "PATH_TO_IMAGES" \
    --prompt "remove degradation" \
    --output_dir "outputs/difix3d+" \
    --timestep 199
```

## Acknowledgements

Our work is built upon the following projects:
- [diffusers](https://github.com/huggingface/diffusers)
- [img2img-turbo](https://github.com/GaParmar/img2img-turbo)
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [DL3DV-10K](https://github.com/DL3DV-10K/Dataset)
- [nerfbusters](https://github.com/ethanweber/nerfbusters)
- [Difix3D+](https://github.com/nv-tlabs/Difix3D)

Shoutout to all the contributors of these projects for their invaluable work that made this research possible.

## License/Terms of Use:

The use of the model and code is governed by the NVIDIA License. See [LICENSE.txt](LICENSE.txt) for details.
Additional Information:  [LICENSE.md · stabilityai/sd-turbo at main](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md)

## Contact
```
ligaoqi02@gmail.com
```