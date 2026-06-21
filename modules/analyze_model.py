import os
import inspect
import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchviz import make_dot
from typing import List, Any, Tuple, Dict, Optional


# --- PyG SHIELD ---
    # PyG DataBatch objects iterate as (str, Tensor) pairs. torchinfo aggressively 
    # iterates over inputs to calculate memory, causing an 'int' + 'str' crash.
    # We wrap the batch to hide its __iter__ method from torchinfo, while allowing
    # the model's forward pass to access attributes via __getattr__ duck-typing.
class PyGShield:
        def __init__(self, batch):
            self._batch = batch
            
        def __getattr__(self, name):
            return getattr(self._batch, name)
            
        # Mock methods to satisfy torchinfo's internal memory/size calculation loops
        def size(self): return (1,)
        def numel(self): return 1
        def element_size(self): return 1
        def dim(self): return 1
        
        # Prevent "unsupported operand type(s) for +: 'int' and 'PyGShield'" when torchinfo uses sum()
        def __add__(self, other):
            return other if isinstance(other, (int, float)) else self
            
        def __radd__(self, other):
            return other if isinstance(other, (int, float)) else self


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
    if args:
        # Wrap any PyG Data/Batch objects to prevent torchinfo from crashing
        safe_args = tuple(PyGShield(a) if hasattr(a, 'edge_index') else a for a in args)
        input_data = safe_args
        additional_kwargs = kwargs
    else:
        # Convert kwargs to a tuple of positional arguments (Tensors) 
        # to prevent torchinfo from crashing on dictionary string keys.
        forward_params = inspect.signature(model.forward).parameters
        positional_args = []
        for name in forward_params:
            if name in kwargs:
                positional_args.append(kwargs[name])
        
        input_data = tuple(positional_args)
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
        return str(model_summary)
    except Exception as e:
        # Fallback if PyG objects or complex shapes cause formatting errors inside torchinfo
        try:
            fallback_summary = summary(
                model,
                input_data=input_data,
                **additional_kwargs,
                col_names=["num_params"], # Exclude size columns to bypass shape formatting completely
                depth=4,
                verbose=0
            )
            print(fallback_summary)
            print("\n")
            return str(fallback_summary)
        except Exception as inner_e:
            # Ultimate fallback to ensure execution never stops and TensorBoard gets a string
            print(f"Torchinfo could not parse the PyG object (Reason: {inner_e}).")
            print("Falling back to standard PyTorch model representation...\n")
            fallback_str = repr(model)
            print(fallback_str)
            print("\n")
            return fallback_str


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


def log_metrics_to_tensorboard(
        train_maes: List[float], 
        val_maes: List[float], 
        model_name: str, 
        log_dir: str,
        train_ewts: Optional[List[float]] = None,
        val_ewts: Optional[List[float]] = None,
        model: Optional[torch.nn.Module] = None,
        model_summary_str: str = ""
        ) -> None:
    """
    Retroactively writes MAE, EwT, Weight Histograms, and Model Summary to TensorBoard.
    """
    print(f">>> 4. Writing to TensorBoard at '{log_dir}' <<<\n")
    tb_path = os.path.join(log_dir, model_name)
    writer = SummaryWriter(log_dir=tb_path)

    # A. Log MAE and EwT Metrics over Epochs
    if train_maes and val_maes:
        max_epochs = min(len(train_maes), len(val_maes))
        for epoch in range(max_epochs):
            # Log MAE
            writer.add_scalars('Metrics/MAE (Mean Absolute Error)', {
                'Train': train_maes[epoch],
                'Validation': val_maes[epoch]
            }, epoch)
            
            # Log EwT if provided
            if train_ewts and val_ewts and epoch < len(train_ewts) and epoch < len(val_ewts):
                writer.add_scalars('Metrics/EwT (Energy within Threshold %)', {
                    'Train': train_ewts[epoch],
                    'Validation': val_ewts[epoch]
                }, epoch)
                
        print(f">> Logged {max_epochs} epochs of MAE and EwT metrics.")
    else:
        print(">> No MAE metrics provided to log.")

    # B. Log Model Weights as Histograms
    if model is not None:
        try:
            for name, param in model.named_parameters():
                if param.requires_grad and param.numel() > 0:
                    # Write the histogram of the tensor values
                    writer.add_histogram(f"Weights/{name}", param.detach().cpu().numpy(), global_step=0)
            print(">> Logged Model Weight Histograms (Check the 'Histograms' tab).")
        except Exception as e:
            print(f">> Could not log weight histograms: {e}")

    # C. Log Model Summary as Text
    if model_summary_str:
        # Wrap the summary in markdown code blocks so it preserves spacing in TensorBoard
        formatted_text = f"```text\n{model_summary_str}\n```"
        writer.add_text("Architecture/Torchinfo_Summary", formatted_text, global_step=0)
        print(">> Logged Model Summary Text (Check the 'Text' tab).")
        
    writer.close()
    print(f"\nTensorBoard writing complete. View with: tensorboard --logdir={log_dir}\n")
