import torch
import numpy as np
import onnxruntime as ort


def _infer_output_names(model, sample_inputs):
    with torch.no_grad():
        test_out = model(*sample_inputs)
        if isinstance(test_out, torch.Tensor):
            return ["output_0"]
        elif isinstance(test_out, (tuple, list)):
            return [f"output_{i}" for i in range(len(test_out))]
        elif isinstance(test_out, dict):
            return list(test_out.keys())


def _export(model, sample_inputs, output_path,
           input_names=None, output_names=None,
           dynamic_axes=None, opset_version=17,
           verbose=True):
           
    if isinstance(sample_inputs, torch.Tensor):
        sample_inputs = (sample_inputs,)

    input_names = input_names or [
        f"input_{i}" for i in range(len(sample_inputs))
    ]

    model.eval()

    if output_names is None:
        output_names = _infer_output_names(model, sample_inputs)

    torch.onnx.export(
        model,
        sample_inputs if len(sample_inputs) > 1 else sample_inputs[0],
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )

    if verbose:
        print(f"exported to {output_path}")

    return input_names, output_names


def _test(model, sample_inputs, output_path,
         input_names, output_names,
         rtol=1e-4, atol=1e-5,
         verbose=True):
    if isinstance(sample_inputs, torch.Tensor):
        sample_inputs = (sample_inputs,)

    model.eval()

    with torch.no_grad():
        torch_out = model(*sample_inputs)

    if isinstance(torch_out, torch.Tensor):
        torch_out = [torch_out]
    elif isinstance(torch_out, dict):
        torch_out = list(torch_out.values())
    elif isinstance(torch_out, tuple):
        torch_out = list(torch_out)

    session = ort.InferenceSession(output_path)
    ort_inputs = {
        name: inp.numpy()
        for name, inp in zip(input_names, sample_inputs)
    }
    ort_out = session.run(None, ort_inputs)

    for i, (t, o) in enumerate(zip(torch_out, ort_out)):
        t_np = t.numpy()
        if not np.allclose(t_np, o, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(t_np - o))
            raise ValueError(
                f"output {output_names[i]} mismatch: "
                f"max diff {max_diff:.6e} (rtol={rtol}, atol={atol})"
            )

    if verbose:
        print(
            f"ONNX validation passed: {len(ort_out)} output(s), "
            f"all within tolerance"
        )


def save(model, sample_inputs, output_path, **kwargs):
    input_names, output_names = _export(
        model, sample_inputs, output_path, **kwargs
    )
    _test(model, sample_inputs, output_path, input_names, output_names)
    print("save complete")
