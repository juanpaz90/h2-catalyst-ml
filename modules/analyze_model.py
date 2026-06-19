import os
import inspect
import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchviz import make_dot
from typing import List, Any, Tuple, Dict


def _get_model_args_kwargs(model: torch.nn.Module, sample_batch: Any) -> Tuple[tuple, dict]:
    """
    Intelligently inspects the model's forward signature to determine how to pass the batch.
    Handles both custom models (expecting a single 'batch' object) and PyG baseline 
    models (expecting unpacked 'z', 'pos', 'batch').
    """
    forward_params = inspect.signature(model.forward).parameters
    
    # 1. Check if the model explicitly asks for a single 'batch' argument (e.g. Custom MHA Model)
    if 'batch' in forward_params and len(forward_params) == 1:
        # Force requires_grad=True on pos to ensure Autograd tracing works for Force predictions
        if hasattr(sample_batch, 'pos') and not sample_batch.pos.requires_grad:
            sample_batch.pos.requires_grad_(True)
        return (sample_batch,), {}

    # 2. Otherwise, unpack attributes for Baseline models (e.g. PyG SchNet)
    kwargs = {}
    
    # Map atomic numbers safely
    if hasattr(sample_batch, 'z'):
        kwargs['z'] = sample_batch.z
    elif hasattr(sample_batch, 'atomic_numbers'):
        kwargs['z'] = sample_batch.atomic_numbers
        
    # Map positions and force grad tracking
    if hasattr(sample_batch, 'pos'):
        kwargs['pos'] = sample_batch.pos
        if not kwargs['pos'].requires_grad:
            kwargs['pos'].requires_grad_(True)
            
    # Map batch index
    if hasattr(sample_batch, 'batch'):
        kwargs['batch'] = sample_batch.batch
        
    # Strictly filter against the model's explicit forward parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in forward_params}
    
    return (), filtered_kwargs


def inspect_model_weights(model: torch.nn.Module) -> None:
    """
    Extracts and prints the shapes of the model's weight tensors.
    """
    print(">>> 1. Extracting Weights <<<\n")
    state_dict = model.state_dict()
    print(f"Total number of weight tensors: {len(state_dict)}")
    
    for i, (layer_name, weight_tensor) in enumerate(state_dict.items()):
        print(f"  - {layer_name}: {weight_tensor.shape}")


def generate_layer_summary(model: torch.nn.Module, sample_batch: Any) -> None:
    """
    Generates a detailed layer-by-layer summary of the model using torchinfo.
    """
    print(">>> 2. Generating Model Summary <<<\n")
    args, kwargs = _get_model_args_kwargs(model, sample_batch)
    
    # torchinfo requires 'input_data' to trace shapes. 
    # It accepts either a tuple (for positional args) or a dict (for keyword args).
    if args:
        input_data = args
        additional_kwargs = kwargs
    else:
        input_data = kwargs
        additional_kwargs = {}

    try:
        model_summary = summary(
            model, 
            input_data=input_data,
            **additional_kwargs,
            col_names=["input_size", "output_size", "num_params"],
            depth=4,
            verbose=0
        )
        print(model_summary)
        print("\n")
    except Exception as e:
        print(f"Could not generate torchinfo summary due to input format: {e}\n")


def export_computational_graph(model: torch.nn.Module, sample_batch: Any, model_name: str) -> None:
    """
    Traces the forward pass and generates a visual graph of the neural network architecture.
    Saves the output as a raw .dot text file to bypass the need for a system-level 
    Graphviz executable installation.
    
    Args:
        model (torch.nn.Module): The loaded PyTorch model.
        sample_batch (Any): A sample input batch to push through the forward pass.
        model_name (str): The base name for the output file.
    """
    print(">>> 3. Generating Computational Graph Graphic <<<\n")
    try:
        # Dynamically fetch safe inputs for this specific model
        args, kwargs = _get_model_args_kwargs(model, sample_batch)
        predictions = model(*args, **kwargs)

        # Isolate the output tensor for autograd tracing
        out_tensor = predictions[0] if isinstance(predictions, tuple) else predictions
        
        # Render the architecture graph blueprint
        graph = make_dot(out_tensor, params=dict(list(model.named_parameters())))
        dot_filename = f"/home/jepazminoc/h2-catalyst-ml/NetworkArchitecture/{model_name}_architecture.dot"
        
        # Save ONLY the text file
        graph.save(dot_filename)
        
        print(f"Saved network raw DOT code to: {dot_filename}")
        
    except Exception as e:
        error_str = str(e)
        if "pyg-lib" in error_str:
            print("--- PYG-LIB DEPENDENCY ERROR ---")
            print("To fix this, please install pyg-lib in your notebook.")
        else:
            print(f"Could not generate graphic: {e}\n")


def log_metrics_to_tensorboard(train_metrics: List[float], val_metrics: List[float], model_name: str, log_dir: str) -> None:
    """
    Retroactively writes training and validation metrics to TensorBoard.
    """
    print(f">>> 4. Writing to TensorBoard at '{log_dir}' <<<\n")
    tb_path = os.path.join(log_dir, model_name)
    writer = SummaryWriter(log_dir=tb_path)

    if train_metrics and val_metrics:
        max_epochs = min(len(train_metrics), len(val_metrics))
        for epoch in range(max_epochs):
            writer.add_scalars('MAE', {
                'Train': train_metrics[epoch],
                'Validation': val_metrics[epoch]
            }, epoch)
        print(f"  - Retroactively logged {max_epochs} epochs of MAE metrics.")
    else:
        print("  - No metrics provided to log.")
        
    writer.close()
    print(f"TensorBoard writing complete. View with: tensorboard --logdir={log_dir}\n")
