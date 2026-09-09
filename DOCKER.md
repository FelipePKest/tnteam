# NVIDIA Docker training

The image matches the working Python 3.10, PyTorch 1.13.1, and CUDA 11.7
environment. StarCraft II, the reference MARIE checkout, and uncontrolled-agent
checkpoints are mounted at runtime and are not copied into the image.

## Host prerequisites

- NVIDIA driver compatible with CUDA 11.7
- Docker Engine and NVIDIA Container Toolkit
- Docker Compose and the NVIDIA runtime
- StarCraft II under `./3rdparty/StarCraftII`
- MARIE checkout at `../MARIE`
- pretrained teammates under `./uncntrl_agents`

Verify GPU passthrough:

```bash
docker run --rm --gpus all nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

Build from the tnteam directory:

```bash
docker-compose -f docker-compose.gpu.yml build
```

Run either experiment in the foreground:

```bash
docker-compose -f docker-compose.gpu.yml run --rm marie
docker-compose -f docker-compose.gpu.yml run --rm poam
```

Run detached and follow its logs:

```bash
docker-compose -f docker-compose.gpu.yml up -d marie
docker-compose -f docker-compose.gpu.yml logs -f marie
```

Use `poam` in place of `marie` for POAM. Avoid starting both together on one
GPU when fastest single-run throughput is the priority; the learners compete
for GPU compute. Results and checkpoints persist in `./naht_results` and
`./3sv5z` on the host.

Override asset locations when they are elsewhere:

```bash
SC2_PATH=/data/StarCraftII \
MARIE_PATH=/repos/MARIE \
UNCONTROLLED_PATH=/data/uncntrl_agents \
docker-compose -f docker-compose.gpu.yml run --rm marie
```
