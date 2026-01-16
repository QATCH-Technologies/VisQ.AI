import torch
import torch.onnx
from src.models import Model  # Ensure this is importable

# 1. Define dummy configuration matching your real data
# These don't need to be exact, just consistent shapes
dummy_batch_size = 1
dummy_cat_maps = {
    "Protein_class_type": ["A", "B"],
    "Regime": ["R1", "R2"],
    "Excipient_Type_1": ["E1", "E2"],
}
dummy_numeric_dim = 10
# Mock split indices: 'conc_1' is at index 0 to 1
dummy_split_indices = {"conc_1": (0, 1)}

# 2. Instantiate the model
model = Model(
    cat_maps=dummy_cat_maps,
    numeric_dim=dummy_numeric_dim,
    out_dim=1,
    hidden_sizes=[32, 32],
    dropout=0.1,
    split_indices=dummy_split_indices,
)

# 3. Create dummy inputs
# Random numerical data
x_num = torch.randn(dummy_batch_size, dummy_numeric_dim)
# Random categorical indices (within vocab size)
x_cat = torch.randint(0, 2, (dummy_batch_size, len(dummy_cat_maps)))

# 4. Export to ONNX
output_path = "visqai_model.onnx"
torch.onnx.export(
    model,
    (x_num, x_cat),
    output_path,
    export_params=True,  # Store the trained parameter weights inside the model file
    opset_version=11,  # The ONNX version to export the model to
    do_constant_folding=True,  # Whether to execute constant folding for optimization
    input_names=["x_num", "x_cat"],  # the model's input names
    output_names=["viscosity_pred"],  # the model's output names
    dynamic_axes={
        "x_num": {0: "batch_size"},  # variable length axes
        "x_cat": {0: "batch_size"},
        "viscosity_pred": {0: "batch_size"},
    },
)

print(f"Model exported to {output_path}")

from torchview import draw_graph

model_graph = draw_graph(
    model,
    input_data=(x_num, x_cat),
    expand_nested=True,  # Shows details inside ResidualBlocks
    graph_name="VisQAI_Architecture",
    depth=2,  # Adjust depth to hide/show layer internals
)

model_graph.visual_graph.render(format="png")
