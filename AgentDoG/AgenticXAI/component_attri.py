import json
import torch
import numpy as np
import os
import math
import argparse
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM


class AttributionMetricCalculator:
    
    @staticmethod
    def calculate_current_state(sum_log_prob, token_count):
        if token_count == 0:
            avg_log_prob = -9999.0
        else:
            avg_log_prob = sum_log_prob / token_count
        try:
            raw_prob = math.exp(avg_log_prob)
        except OverflowError:
            raw_prob = 0.0
        return {
            "sum_log_prob": sum_log_prob,
            "token_count": token_count,
            "avg_log_prob": avg_log_prob,
            "raw_prob": raw_prob
        }

    @staticmethod
    def calculate_differentials(current_metrics, prev_metrics):
        if prev_metrics is None:
            return {
                "prob_diff": 0.0,
                "llr_score": 0.0
            }
        prob_diff = current_metrics["raw_prob"] - prev_metrics["raw_prob"]
        llr_score = current_metrics["avg_log_prob"] - prev_metrics["avg_log_prob"]
        return {
            "prob_diff": prob_diff,
            "llr_score": llr_score
        }


class TrajectoryModelWrapper:
    ROLE_MAPPING = {
        "user": "user",
        "human": "user",
        "assistant": "assistant",
        "agent": "assistant",
        "environment": "user",
        "tool": "user",
        "system": "system"
    }

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _normalize_message(self, message):
        if not isinstance(message, dict):
            return {"role": "user", "content": str(message)}
        role = message.get("role", "user")
        if isinstance(role, str):
            role = role.lower()
        mapped_role = self.ROLE_MAPPING.get(role, "user")
        content = message.get("content", "")
        if isinstance(content, list):
            pieces = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block and block["text"] is not None:
                        pieces.append(str(block["text"]))
                    elif "content" in block and block["content"] is not None:
                        pieces.append(str(block["content"]))
                elif block is not None:
                    pieces.append(str(block))
            content = "\n".join(pieces)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        return {"role": mapped_role, "content": content}

    def _normalize_messages(self, messages):
        return [self._normalize_message(msg) for msg in messages]

    def get_generation_confidence(self, context_messages, target_message):
        normalized_context = self._normalize_messages(context_messages)
        normalized_target = self._normalize_message(target_message)
        full_conversation = normalized_context + [normalized_target]
        input_ids = self.tokenizer.apply_chat_template(
            full_conversation,
            return_tensors="pt",
            add_generation_prompt=False
        ).to(device=self.model.device, dtype=torch.long)
        if len(normalized_context) > 0:
            context_ids = self.tokenizer.apply_chat_template(
                normalized_context,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(device=self.model.device, dtype=torch.long)
            context_len = context_ids.shape[1]
        else:
            try:
                empty_ids = self.tokenizer.apply_chat_template([], return_tensors="pt", add_generation_prompt=True)
                context_len = empty_ids.shape[1] if empty_ids is not None else 0
            except:
                context_len = 0
        target_ids = input_ids.clone()
        context_len = min(context_len, target_ids.shape[1])
        target_ids[:, :context_len] = -100
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)
            sum_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            token_count = (shift_labels != -100).sum().item()
            sum_log_prob = -sum_loss.item()
        return sum_log_prob, token_count


class AnalysisPipeline:
    def __init__(self, model_wrapper, args=None):
        self.wrapper = model_wrapper
        self.calculator = AttributionMetricCalculator()
        self.args = args

    def run(self, trajectory_data, filename="unknown"):
        if 'trajectory' in trajectory_data:
            full_traj = trajectory_data['trajectory']
        elif 'trace' in trajectory_data:
            full_traj = trajectory_data['trace']
        else:
            print(f"Skipping {filename}: No 'trajectory' or 'trace' field found.")
            return None
        if len(full_traj) < 1:
            print(f"Skipping {filename}: Trajectory empty.")
            return None
        print('file_name', filename)
        target_message = full_traj[-1]
        query_message = full_traj[-2]
        if 'memory_benign_weather' in filename.lower() or 'memory_benign_sport_split' in filename.lower():
            query_message = full_traj[-3]
        history_events = full_traj[:-1]
        if 'memory' in filename.lower():
            history_events = full_traj[:-2]
        results = []
        prev_metrics = None
        print(f"\nAnalyzing File: {filename}")
        print(f"Target Role: {target_message.get('role', 'assistant')} | History Length: {len(history_events)} steps")
        print(f"Target content: {target_message}")
        print("-" * 110)
        print(f"{'Step':<5} | {'Role':<10} | {'Avg LogProb':<12} | {'Raw Prob':<12} | {'Prob Diff':<12} | {'LLR Score':<12}")
        print("-" * 110)
        for k in range(len(history_events) + 1):
            context_for_step = history_events[:k]
            if 'memory' in filename.lower():
                context_for_step = history_events[:k] + [query_message]
            print('context_for_step:', context_for_step)
            sum_log_prob, token_count = self.wrapper.get_generation_confidence(
                context_for_step, target_message
            )
            curr_metrics = self.calculator.calculate_current_state(sum_log_prob, token_count)
            diff_metrics = self.calculator.calculate_differentials(curr_metrics, prev_metrics)
            combined_metrics = {**curr_metrics, **diff_metrics}
            if k == 0:
                step_role = "START"
                step_content = "Empty Context"
            else:
                step_role = history_events[k-1]['role']
                step_content = history_events[k-1]['content'][:50].replace('\n', ' ') + "..."
            step_data = {
                "step_index": k,
                "event_role": step_role,
                "event_content_preview": step_content,
                "metrics": combined_metrics
            }
            results.append(step_data)
            prev_metrics = curr_metrics
            print(f"{k:<5} | {step_role:<10} | "
                  f"{combined_metrics['avg_log_prob']:<12.4f} | "
                  f"{combined_metrics['raw_prob']:<12.2%} | "
                  f"{combined_metrics['prob_diff']:<+12.2%} | "
                  f"{combined_metrics['llr_score']:<+12.4f}")
            print("-" * 110)
        return results


def parse_arguments():
    parser = argparse.ArgumentParser(description="Agent Attribution XAI Analysis Tool")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        return
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading Tokenizer from {args.model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Loading Model from {args.model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    model_wrapper = TrajectoryModelWrapper(model, tokenizer)
    pipeline = AnalysisPipeline(model_wrapper, args=args)
    json_files = glob.glob(os.path.join(args.data_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {args.data_dir}")
        return
    print(f"Found {len(json_files)} files to process.")
    for file_path in json_files:
        try:
            file_name = os.path.basename(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {file_name}, skipping.")
                continue
            analysis_results = pipeline.run(data, filename=file_name)
            if analysis_results:
                model_name_safe = args.model_id.strip("/").split("/")[-1]
                base_name = os.path.splitext(file_name)[0]
                output_filename = f"{base_name}_{model_name_safe}_attr_trajectory.json"
                output_path = os.path.join(args.output_dir, output_filename)
                output = {
                    "meta_info": {
                        "model": args.model_id,
                        "source_file": file_name,
                        "data_id": data.get("data_id", "unknown"),
                        "target_response_preview": (data.get('trajectory') or data.get('trace') or [{'content': ''}])[-1]['content'][:100]
                    },
                    "trajectory_analysis": analysis_results
                }
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"Saved results to: {output_path}")
        except:
            continue
    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()