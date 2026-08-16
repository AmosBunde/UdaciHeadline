import torch
import torch.nn as nn
import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.accelerator import get_accelerator
import argparse
import time
import os

# --- Model Definition ---
class MockTransformerBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

class RealisticModel(nn.Module):
    def __init__(self, hidden_size=2048, num_layers=30, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # ## STUDENT TASK 1: Fix the Model Architecture ##
        # nn.ModuleList has no forward() and is treated by DeepSpeed's "uniform" partitioner as ONE opaque
        # block, so it cannot be split. nn.Sequential exposes the blocks as an ordered, callable sequence
        # of layers, which the partitioner can cut anywhere between two blocks.
        self.transformer_blocks = nn.Sequential(
            *[MockTransformerBlock(hidden_size) for _ in range(num_layers)]
        )

        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        # nn.Sequential is callable: it applies every block in order
        x = self.transformer_blocks(x)
        x = self.output_head(x)
        return x

    def to_layers(self):
        """Flat list of layers for DeepSpeed's PipelineModule (each entry can live on a different stage)."""
        return [self.embedding, *self.transformer_blocks, self.output_head]

# --- Helper Function for Performance Measurement ---
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def measure_throughput(model, dummy_input, iterations):
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    _sync()

    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    _sync()
    end_time = time.time()

    total_samples = dummy_input.size(0) * iterations
    duration = end_time - start_time
    throughput = total_samples / duration
    return throughput

def measure_pipeline_throughput(engine, batch_fn, iterations):
    """Throughput of a DeepSpeed PipelineEngine: eval_batch() pushes one *global* batch through the
    pipeline as `train_batch_size / micro_batch_size` micro-batches (the 1F schedule)."""
    def data_iter():
        while True:
            yield batch_fn()
    for _ in range(3):                       # warm-up
        engine.eval_batch(data_iter(), compute_loss=False, reduce_output=None)
    _sync()
    torch.distributed.barrier()
    start_time = time.time()
    for _ in range(iterations):
        engine.eval_batch(data_iter(), compute_loss=False, reduce_output=None)
    _sync()
    torch.distributed.barrier()
    duration = time.time() - start_time
    return engine.train_batch_size() * iterations / duration

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    # Problem size knobs. Defaults reproduce the exercise on GPUs; the notebook passes smaller values on CPU.
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=30)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # --- Setup ---
    global_batch_size = 64
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    vocab_size = 1000
    seq_len = args.seq_len
    iterations = args.iterations
    is_rank_0 = args.local_rank <= 0
    device_name = get_accelerator().device_name()          # "cuda" or "cpu"
    first_device = f"{device_name}:0" if device_name == "cuda" else "cpu"

    # --- Baseline Measurement (Single GPU) ---
    if is_rank_0:
        print(f"\n--- Measuring Baseline Performance (Single device: {first_device}) ---", flush=True)
        # ## STUDENT TASK 2: Implement the Baseline Measurement ##
        try:
            baseline_model = RealisticModel(hidden_size, num_layers, vocab_size).to(first_device).eval()
            dummy_input = torch.randint(0, vocab_size, (global_batch_size, seq_len), device=first_device)
            baseline_throughput = measure_throughput(baseline_model, dummy_input, iterations)
            print(f"Baseline Throughput: {baseline_throughput:.2f} samples/sec", flush=True)
            del baseline_model, dummy_input
        except Exception as e:
            print(f"Could not run baseline on a single device: {e}", flush=True)
        print("--------------------------------------------------\n", flush=True)

    # --- DeepSpeed Pipeline Parallelism ---
    # deepspeed.init_distributed picks NCCL on GPU and gloo on CPU
    deepspeed.init_distributed(dist_backend="nccl" if device_name == "cuda" else "gloo")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    world_size = torch.distributed.get_world_size()
    print(f"\n--- Rank {args.local_rank}: Setting up DeepSpeed Pipeline (world size {world_size}) ---", flush=True)

    ds_model = RealisticModel(hidden_size, num_layers, vocab_size)
    # PipelineModule cuts the flat layer list into `num_stages` contiguous chunks ("uniform" = equal
    # number of layers per stage) and places each chunk on the rank that owns that stage.
    pipe_model = PipelineModule(layers=ds_model.to_layers(), num_stages=world_size,
                                partition_method="uniform", loss_fn=None)

    # ## STUDENT TASK 3: Initialize the DeepSpeed Engine ##
    model_engine, _, _, _ = deepspeed.initialize(args=args, model=pipe_model,
                                                 model_parameters=[p for p in pipe_model.parameters() if p.requires_grad])
    model_engine.eval()

    # Input needs to be integer indices for the embedding layer. eval_batch expects (inputs, labels) tuples.
    def make_batch():
        return (torch.randint(0, vocab_size, (global_batch_size, seq_len), device=model_engine.device),
                torch.zeros(global_batch_size, dtype=torch.long, device=model_engine.device))

    pipelined_throughput = measure_pipeline_throughput(model_engine, make_batch, iterations)

    if model_engine.is_last_stage():
        print(f"\n--- Results on Last Stage (Rank {args.local_rank}) ---", flush=True)
        print(f"micro_batch_size={model_engine.train_micro_batch_size_per_gpu()}  stages={world_size}", flush=True)
        print(f"Pipelined Throughput: {pipelined_throughput:.2f} samples/sec", flush=True)
        print("------------------------------------------\n", flush=True)

if __name__ == "__main__":
    main()
