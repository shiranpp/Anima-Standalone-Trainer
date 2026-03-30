
import argparse
import os
import sys
import torch
import threading
import copy
from accelerate import Accelerator
from library import lumina_util, lumina_train_util, strategy_lumina
from library.strategy_base import TokenizeStrategy, TextEncodingStrategy
from library.flux_models import AutoEncoder
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler
from library.utils import setup_logging
from library.device_utils import clean_memory_on_device

setup_logging()
import logging
logger = logging.getLogger(__name__)

# Global models cache for server mode
CACHED_MODELS = None
APP_ACCELERATOR = None
CURRENT_LORA = {"path": None, "mul": 1.0}

def _apply_lora(accelerator, models, lora_path, multiplier):
    import networks.lora_lumina
    from safetensors.torch import load_file
    weights_sd = load_file(lora_path)
    net, _ = networks.lora_lumina.create_network_from_weights(
        multiplier=multiplier,
        file=None,
        ae=models["ae"],
        text_encoders=[models["gemma2"]],
        unet=models["dit"],
        weights_sd=weights_sd,
        for_inference=True
    )
    net.merge_to([models["gemma2"]], models["dit"], weights_sd)
    
    # Sync with secondary model if using Parallel CFG
    if models.get("dit_secondary") is not None:
        sec_device = next(models["dit_secondary"].parameters()).device
        net.merge_to([], models["dit_secondary"], weights_sd)
        # Handle potential dtype mismatches if necessary
        models["dit_secondary"].to(sec_device)
    del net, weights_sd
    torch.cuda.empty_cache()

def manage_lora(accelerator, models, target_path, target_mul):
    global CURRENT_LORA

    if CURRENT_LORA["path"] == target_path and abs(CURRENT_LORA["mul"] - target_mul) < 1e-6:
        return

    if CURRENT_LORA["path"]:
        logger.info(f"Unmerging previous LoRA: {CURRENT_LORA['path']}")
        try:
            _apply_lora(accelerator, models, CURRENT_LORA["path"], -CURRENT_LORA["mul"])
            CURRENT_LORA["path"] = None
        except Exception as e:
            logger.error(f"Failed to unmerge LoRA: {e}")

    if target_path:
        logger.info(f"Merging new LoRA: {target_path} (x{target_mul})")
        try:
            _apply_lora(accelerator, models, target_path, target_mul)
            CURRENT_LORA["path"] = target_path
            CURRENT_LORA["mul"] = target_mul
        except Exception as e:
            logger.error(f"Failed to merge LoRA: {e}")


def load_models(args, accelerator):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load DiT
    logger.info(f"Loading Lumina DiT from {args.dit_path}")
    dit = lumina_util.load_lumina_model(
        args.dit_path,
        dtype=weight_dtype,
        device="cpu",
        use_flash_attn=getattr(args, 'flash_attn', False),
        use_sage_attn=getattr(args, 'sage_attn', False),
    )
    dit.eval()

    # Load Gemma2
    logger.info(f"Loading Gemma2 from {args.gemma2}")
    gemma2 = lumina_util.load_gemma2(args.gemma2, dtype=weight_dtype, device="cpu")
    gemma2.eval()

    # Load AE (VAE)
    logger.info(f"Loading AE from {args.ae}")
    ae = lumina_util.load_ae(args.ae, dtype=weight_dtype, device="cpu")
    ae.eval()

    # Load LoRA if specified
    if args.network_weights:
        logger.info(f"Loading LoRA weights from {args.network_weights}")
        import networks.lora_lumina
        from safetensors.torch import load_file
        weights_sd = load_file(args.network_weights)
        network, _ = networks.lora_lumina.create_network_from_weights(
            multiplier=args.network_mul,
            file=None,
            ae=ae,
            text_encoders=[gemma2],
            unet=dit,
            weights_sd=weights_sd,
            for_inference=True
        )
        network.merge_to([gemma2], dit, weights_sd)
        logger.info(f"LoRA merged with multiplier {args.network_mul}")

        global CURRENT_LORA
        CURRENT_LORA["path"] = args.network_weights
        CURRENT_LORA["mul"] = args.network_mul

    # Distribute models across GPUs
    dit_secondary = None
    if args.device_map == 'parallel_cfg' and torch.cuda.device_count() > 1:
        logger.info("Parallel CFG mode: loading model on both GPUs")
        dit.to(torch.device('cuda:0'))
        
        dit_secondary = copy.deepcopy(dit)
        dit_secondary.to(torch.device('cuda:1'))
        logger.info("  Primary model on GPU 0, Secondary on GPU 1")
        
        gemma2.to(torch.device('cuda:0'))
        ae.to("cpu")
    elif args.device_map == 'sharding' and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Multi-GPU sharding: distributing model across {num_gpus} GPUs")
        
        vram = [torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)]
        total_vram = sum(vram)
        num_layers = len(dit.layers)
        
        # Split layers proportionally
        gpu_assignments = []
        layers_assigned = 0
        for i in range(num_gpus):
            if i == num_gpus - 1:
                n = num_layers - layers_assigned
            else:
                n = round(num_layers * vram[i] / total_vram)
            gpu_assignments.append(n)
            layers_assigned += n

        first_gpu = torch.device('cuda:0')
        last_gpu = torch.device(f'cuda:{num_gpus - 1}')
        
        # Move base components
        dit.x_embedder.to(first_gpu)
        dit.t_embedder.to(first_gpu)
        dit.cap_embedder.to(first_gpu)
        if hasattr(dit, 'noise_refiner'): dit.noise_refiner.to(first_gpu)
        if hasattr(dit, 'context_refiner'): dit.context_refiner.to(first_gpu)
        
        dit.norm_final.to(last_gpu)
        dit.final_layer.to(last_gpu)
        
        # Move layers
        layer_idx = 0
        for gpu_id in range(num_gpus):
            device = torch.device(f'cuda:{gpu_id}')
            for _ in range(gpu_assignments[gpu_id]):
                dit.layers[layer_idx].to(device)
                layer_idx += 1
            logger.info(f"  GPU {gpu_id}: {gpu_assignments[gpu_id]} layers")
            
        dit._is_multi_gpu_sharded = True
        gemma2.to(first_gpu)
        ae.to("cpu")
    else:
        dit.to(accelerator.device)
        gemma2.to(accelerator.device)
        ae.to("cpu")  # Keep AE on CPU, move to GPU only during decode

    tokenize_strategy = strategy_lumina.LuminaTokenizeStrategy(
        getattr(args, 'system_prompt', ''),
        getattr(args, 'gemma2_max_token_length', 256)
    )
    text_encoding_strategy = strategy_lumina.LuminaTextEncodingStrategy()

    # Register strategies globally for lumina_train_util usage
    if TokenizeStrategy.get_strategy() is None:
        TokenizeStrategy.set_strategy(tokenize_strategy)
    if TextEncodingStrategy.get_strategy() is None:
        TextEncodingStrategy.set_strategy(text_encoding_strategy)

    return {
        "dit": dit,
        "dit_secondary": dit_secondary,
        "gemma2": gemma2,
        "ae": ae,
        "tokenize_strategy": tokenize_strategy,
        "text_encoding_strategy": text_encoding_strategy,
        "dtype": weight_dtype
    }


def perform_generation(args, models, accelerator):
    logger.info("Starting Lumina generation...")

    lumina_train_util.sample_images(
        accelerator, args, 0, 0,
        models["dit"],
        models["ae"],
        models["gemma2"],
        None,  # sample_prompts_gemma2_outputs
        None,  # prompt_replacement
        None,  # controlnet
        dit_secondary=models.get("dit_secondary")
    )
    logger.info("Generation finished.")


def run_server(args, accelerator):
    global CACHED_MODELS
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        logger.error("Flask is required for server mode. Please install it.")
        sys.exit(1)

    app = Flask(__name__)

    CACHED_MODELS = load_models(args, accelerator)
    logger.info("Models loaded. Server ready.")

    gen_lock = threading.Lock()

    @app.route('/generate', methods=['POST'])
    def handle_generate():
        if gen_lock.locked():
            return jsonify({"success": False, "error": "Generation already in progress."}), 429

        with gen_lock:
            try:
                data = request.json
                if 'sample_prompts' in data:
                    args.sample_prompts = data['sample_prompts']

                # Handle Dynamic LoRA Switching
                req_weights = data.get('network_weights')
                req_mul = float(data.get('network_mul', 1.0))
                manage_lora(accelerator, CACHED_MODELS, req_weights, req_mul)

                # Handle Dynamic Attention Switching
                req_flash = bool(data.get('flash_attn', False))
                req_sage = bool(data.get('sage_attn', False))
                if 'dit' in CACHED_MODELS:
                    CACHED_MODELS['dit'].set_flash_attn(req_flash) if hasattr(CACHED_MODELS['dit'], 'set_flash_attn') else None
                    CACHED_MODELS['dit'].set_sage_attn(req_sage) if hasattr(CACHED_MODELS['dit'], 'set_sage_attn') else None

                perform_generation(args, CACHED_MODELS, accelerator)
                return jsonify({"success": True})
            except Exception as e:
                import traceback
                logger.error(f"Generation failed: {e}\n{traceback.format_exc()}")
                return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/ping', methods=['GET'])
    def ping():
        return jsonify({"status": "ready"})

    @app.route('/stop', methods=['POST'])
    def stop():
        func = request.environ.get('werkzeug.server.shutdown')
        if func:
            func()
        else:
            os._exit(0)
        return jsonify({"success": True})

    app.run(host='0.0.0.0', port=args.server_port, debug=False, use_reloader=False)


def main():
    parser = argparse.ArgumentParser()

    # Model paths
    parser.add_argument("--dit_path", type=str, required=True, help="Path to Lumina DiT model")
    parser.add_argument("--gemma2", type=str, required=True, help="Path to Gemma2 text encoder")
    parser.add_argument("--ae", type=str, required=True, help="Path to AutoEncoder (VAE)")

    # Sampling
    parser.add_argument("--sample_prompts", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=None)

    # Lumina-specific
    parser.add_argument("--discrete_flow_shift", type=float, default=6.0)
    parser.add_argument("--cfg_trunc_ratio", type=float, default=0.25)
    parser.add_argument("--renorm_cfg", type=float, default=1.0)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--gemma2_max_token_length", type=int, default=256)

    # Attention
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--sage_attn", action="store_true")

    # LoRA
    parser.add_argument("--network_weights", type=str, default=None)
    parser.add_argument("--network_mul", type=float, default=1.0)

    # Multi-GPU
    parser.add_argument("--device_map", type=str, default=None)

    # Server mode
    parser.add_argument("--server_port", type=int, default=None)

    # Fake args expected by sample_images
    parser.add_argument("--sample_at_first", action="store_true", default=True)
    parser.add_argument("--sample_every_n_steps", type=int, default=None)
    parser.add_argument("--sample_every_n_epochs", type=int, default=None)
    parser.add_argument("--sample_batch_size", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=1)

    args = parser.parse_args()
    args.sample_at_first = True

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    if args.server_port:
        logger.info(f"Starting Lumina Generation Server on port {args.server_port}")
        run_server(args, accelerator)
    else:
        models = load_models(args, accelerator)
        perform_generation(args, models, accelerator)

if __name__ == "__main__":
    main()
