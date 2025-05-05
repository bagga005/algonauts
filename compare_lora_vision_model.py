from model_lora_vision import VisionLinearRegressionModel, RegressionHander_Vision
import torch
import pandas as pd


def compare_two_models(model1: torch.nn.Module, model2: torch.nn.Module, filter_learnable: bool = False, show_structure_only: bool = False):
    comparison = []
    
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    
    all_layer_names = sorted(set(params1.keys()) | set(params2.keys()))
    
    for name in all_layer_names:
        param1 = params1.get(name, None)
        param2 = params2.get(name, None)
        
        row = {"Layer": name}
        
        if param1 is None or param2 is None:
            row.update({
                "Num Params": float('nan'),
                "Mean L2 Diff": float('nan'),
                "Std of Diff": float('nan'),
                "Avg Weight (Model1)": float('nan'),
                "Avg Weight (Model2)": float('nan'),
                "Learnable": "Unknown",
                "Note": "Missing in one model"
            })
        else:
            num_params = param1.numel()  # Count number of parameters in the layer
            row["Num Params"] = num_params
            
            if not show_structure_only:
                diff = (param1 - param2).detach().flatten()
                mean_diff = diff.abs().mean().item()
                std_diff = diff.std().item()
                
                avg1 = param1.detach().mean().item()
                avg2 = param2.detach().mean().item()
                learnable = param1.requires_grad and param2.requires_grad
                
                if filter_learnable and not learnable:
                    continue
                
                row.update({
                    "Mean L2 Diff": mean_diff,
                    "Std of Diff": std_diff,
                    "Avg Weight (Model1)": avg1,
                    "Avg Weight (Model2)": avg2,
                    "Learnable": learnable,
                    "Note": ""
                })
        
        comparison.append(row)
    
    df = pd.DataFrame(comparison)
    
    # If show_structure_only is True, only keep Layer and Num Params columns
    if show_structure_only:
        df = df[["Layer", "Num Params"]]
    
    df = df.sort_values("Layer").reset_index(drop=True)
    
    print(df.to_string(index=False))
    
    # Summary
    total_layers = len(df)
    total_params = df["Num Params"].sum()
    
    print("\n--- Summary ---")
    print(f"Total layers shown: {total_layers}")
    print(f"Total parameters: {total_params:,}")
    
    if not show_structure_only:
        nonzero_diff_layers = df["Mean L2 Diff"].fillna(0).apply(lambda x: x != 0).sum()
        percent_nonzero = (nonzero_diff_layers / total_layers * 100) if total_layers > 0 else 0.0
        print(f"Layers with non-zero mean difference: {nonzero_diff_layers}")
        print(f"Percentage of layers with non-zero mean difference: {percent_nonzero:.2f}%")
    
    return df



# Run the comparison with the updated function
if __name__ == "__main__":
    # model1 = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    # model1.load_state_dict(torch.load('/home/bagga005/algo/comp_data/models/no-train_vision.pth'))
    # model2 = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    # model2.load_state_dict(torch.load('/home/bagga005/algo/comp_data/models/lora-5-distributed-s15_vision.pth'))
    v_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    v_model2 = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    # model1 = VisionLinearRegressionModel(input_size=8192*4, output_size=1000, device='cpu')
    # #model1.load_state_dict(torch.load('/home/bagga005/algo/comp_data/models/lora-5-distributed-s15.pth'))
    # model2 = VisionLinearRegressionModel(input_size=8192*4, output_size=1000, device='cpu')
    # model2.load_state_dict(torch.load('/home/bagga005/algo/comp_data/models/lora-20-distributed-s15.pth', map_location=torch.device('cpu')))
    # Compare vision models with all parameters (including base SlowR50 parameters)
    results = compare_two_models(
        v_model, 
        v_model2,
        filter_learnable=False,
        show_structure_only=True
    )

    # rgv = RegressionHander_Vision(input_size=8192*4, output_size=1000, pretrain_params_name='lora-20-distributed-s15')
    # rgv.save_model('lora-20-distributed-s15', separate_vision=True)
    
    # You can also compare just the LoRA parameters
    # print("\n\n======== COMPARING ONLY LORA PARAMETERS ========")
    # results_lora = compare_model_checkpoints(
    #     checkpoint_paths=['/home/bagga005/algo/comp_data/models/no-train_vision.pth', 
    #                     '/home/bagga005/algo/comp_data/models/lora-5-distributed-s15_vision.pth'],
    #     focus_on_lora=True,  # Only include LoRA parameters
    #     include_base_model=False  # Exclude base model parameters
    # )