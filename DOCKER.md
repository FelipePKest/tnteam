# NVIDIA Docker training

The image matches the working Python 3.10, PyTorch 1.13.1, and CUDA 11.7
environment. The MARIE model, learner, controller, replay procedure, and vector
quantizer are all included in tnteam. Only StarCraft II and uncontrolled-agent
checkpoints are mounted at runtime.

## Host prerequisites

- NVIDIA driver compatible with CUDA 11.7
- Docker Engine and NVIDIA Container Toolkit
- Docker Compose and the NVIDIA runtime
- StarCraft II under `./3rdparty/StarCraftII`
- pretrained teammates under `./uncntrl_agents`

Verify GPU passthrough:

```bash
docker run --rm --gpus all nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

Build from the tnteam directory:

```bash
docker-compose -f docker-compose.gpu.yml build
```

After changing dependencies, force Docker to rebuild the installation layer:

```bash
docker build --no-cache -t tnteam-marie:cu117 .
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
UNCONTROLLED_PATH=/data/uncntrl_agents \
docker-compose -f docker-compose.gpu.yml run --rm marie
```

For plain Docker:

```bash
docker run --rm --gpus all \
  --shm-size=8g \
  -v "$(realpath ./3rdparty/StarCraftII):/workspace/tnteam/3rdparty/StarCraftII:ro" \
  -v "$(realpath ./uncntrl_agents):/workspace/tnteam/uncntrl_agents:ro" \
  -v "$(realpath ./naht_results):/workspace/tnteam/naht_results" \
  -v "$(realpath ./3sv5z):/workspace/tnteam/3sv5z" \
  tnteam-marie:cu117 \
  python train_marie_naht_3sv5z.py --steps 500000 --eval-episodes 8 \
    --seed 112358 --batch-size-run 1 --uncontrolled-seed 112358
```

Confirm the image contains the local MARIE implementation before starting a
long run:

```bash
docker run --rm \
  tnteam-marie:cu117 \
  python -c "from modules.marie import MARIEPolicy; print('MARIE import OK')"
```
