import json
import torch
import os
import argparse
import re
import copy
import math
from transformers import AutoTokenizer, AutoModelForCausalLM


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


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
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block and block["text"] is not None:
                        parts.append(str(block["text"]))
                    elif "content" in block and block["content"] is not None:
                        parts.append(str(block["content"]))
                elif block is not None:
                    parts.append(str(block))
            content = "\n".join(parts)
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
            try:
                context_ids = self.tokenizer.apply_chat_template(
                    normalized_context,
                    return_tensors="pt",
                    add_generation_prompt=True
                ).to(device=self.model.device, dtype=torch.long)
                context_len = context_ids.shape[1]
            except Exception:
                context_len = 0
        else:
            try:
                empty_ids = self.tokenizer.apply_chat_template([], return_tensors="pt", add_generation_prompt=True)
                if empty_ids is not None:
                    empty_ids = empty_ids.to(device=self.model.device, dtype=torch.long)
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
            sum_log_prob = -sum_loss.item()
        return sum_log_prob


def split_into_sentences(text):
    if not text:
        return []
    code_map = {}

    def replace_code_with_placeholder(match):
        index = len(code_map)
        placeholder = f"__CODE_BLOCK_{index}__"
        code_map[placeholder] = match.group(0)
        return placeholder

    masked_text = re.sub(r'<code>.*?</code>', replace_code_with_placeholder, text, flags=re.DOTALL)
    parts = re.split(r'([.!?\n]+)', masked_text)
    sentences = []
    current_sent = ""
    for i in range(0, len(parts) - 1, 2):
        content = parts[i]
        delimiter = parts[i + 1]
        if delimiter == '.' and content.strip().isdigit():
            current_sent += content + delimiter
        else:
            full_sent = current_sent + content + delimiter
            if full_sent.strip():
                sentences.append(full_sent)
            current_sent = ""
    if len(parts) % 2 == 1:
        remaining = current_sent + parts[-1]
        if remaining.strip():
            sentences.append(remaining)
    final_sentences = []
    for sent in sentences:
        if "__CODE_BLOCK_" in sent:
            for placeholder, original_code in code_map.items():
                if placeholder in sent:
                    sent = sent.replace(placeholder, original_code)
        final_sentences.append(sent)
    return final_sentences


class SentenceAttributionAnalyzer:
    def __init__(self, model_wrapper, args=None):
        self.wrapper = model_wrapper
        self.args = args

    def analyze_step(self, full_trajectory, step_index, traj_index):
        target_message = full_trajectory[-1]
        target_step = full_trajectory[traj_index]
        prefix_context = full_trajectory[:traj_index]
        if self.args:
            if 'memory' in self.args.traj_file.lower():
                prefix_context = prefix_context + full_trajectory[-2]
        system_context = []
        if len(full_trajectory) > 0 and full_trajectory[0]['role'] == 'system':
            system_context = [full_trajectory[0]]
        original_text = target_step['content']
        sentences = split_into_sentences(original_text)
        if not sentences:
            return None
        results = {
            "step_index": step_index,
            "traj_index": traj_index,
            "role": target_step['role'],
            "original_content": original_text,
            "sentence_analysis": []
        }
        current_context_full = prefix_context + [target_step]
        base_log_prob = self.wrapper.get_generation_confidence(current_context_full, target_message)
        temp_sentence_results = []
        for idx, sent in enumerate(sentences):
            text_removed = "".join([s for j, s in enumerate(sentences) if j != idx])
            step_removed = copy.deepcopy(target_step)
            step_removed['content'] = text_removed
            context_drop = prefix_context + [step_removed]
            drop_log_prob = self.wrapper.get_generation_confidence(context_drop, target_message)
            drop_score = base_log_prob - drop_log_prob
            step_hold = copy.deepcopy(target_step)
            step_hold['content'] = sent
            context_hold = prefix_context + [step_hold]
            hold_log_prob = self.wrapper.get_generation_confidence(context_hold, target_message)
            hold_score = hold_log_prob - base_log_prob
            total_score = drop_score + hold_score
            temp_sentence_results.append({
                "sentence_index": idx,
                "text": sent,
                "scores": {
                    "drop_score": drop_score,
                    "hold_score": hold_score,
                    "total_score": total_score
                }
            })
        results["sentence_analysis"] = temp_sentence_results
        return results


def parse_arguments():
    parser = argparse.ArgumentParser(description="Agent Sentence-Level Attribution")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--attr_file", type=str, required=True)
    parser.add_argument("--traj_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_arguments()
    json_output_path = args.output_file
    print(f"{Colors.HEADER}Loading Model: {args.model_id}...{Colors.ENDC}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    wrapper = TrajectoryModelWrapper(model, tokenizer)
    analyzer = SentenceAttributionAnalyzer(wrapper)
    with open(args.attr_file, 'r') as f:
        attr_data = json.load(f)
    with open(args.traj_file, 'r') as f:
        traj_data = json.load(f)
    full_trajectory = traj_data.get('trajectory') or traj_data.get('trace')
    traj_analysis = attr_data.get("trajectory_analysis", [])
    print(f"\n{Colors.HEADER}=== Trajectory Attribution Overview ==={Colors.ENDC}")
    print(f"{'Attr Idx':<10} | {'Traj Idx':<10} | {'Role':<10} | {'Prob Diff':<12}")
    print("-" * 50)
    valid_steps = []
    for item in traj_analysis:
        attr_idx = item['step_index']
        if attr_idx == 0:
            continue
        traj_idx = attr_idx - 1
        if traj_idx == 0:
            continue
        if traj_idx >= len(full_trajectory):
            continue
        role = full_trajectory[traj_idx]['role']
        llr_score = item['metrics']['llr_score']
        valid_steps.append({
            "attr_index": attr_idx,
            "traj_index": traj_idx,
            "role": role,
            "llr_score": llr_score,
            "full_item": item
        })
    sorted_steps = sorted(valid_steps, key=lambda x: x['llr_score'], reverse=True)
    for s in sorted_steps:
        is_top = s in sorted_steps[:args.top_k]
        color = Colors.GREEN if is_top else ""
        print(f"{color}{s['attr_index']:<10} | {s['traj_index']:<10} | {s['role']:<10} | {s['llr_score']:<12.4f}{Colors.ENDC}")
    top_steps = sorted_steps[:args.top_k]
    print(f"\nAnalyzing Top-{args.top_k} steps: {[s['traj_index'] for s in top_steps]}")
    final_output = {
        "meta_info": attr_data.get("meta_info", {}),
        "sentence_attribution": []
    }
    for step_info in top_steps:
        t_idx = step_info['traj_index']
        a_idx = step_info['attr_index']
        print(f"\n{Colors.HEADER}>>> Analyzing Traj Step {t_idx} (Attr Step {a_idx}) - Role: {step_info['role']}{Colors.ENDC}")
        step_result = analyzer.analyze_step(full_trajectory, a_idx, t_idx)
        if step_result:
            final_output["sentence_attribution"].append(step_result)
            scores = [(s['scores']['total_score'], i) for i, s in enumerate(step_result['sentence_analysis'])]
            scores.sort(key=lambda x: x[0], reverse=True)
            top_3_indices = {idx for score, idx in scores[:3]}
            for sent_data in step_result['sentence_analysis']:
                idx = sent_data['sentence_index']
                total = sent_data['scores']['total_score']
                text_preview = sent_data['text'][:60].replace('\n', ' ') + "..."
                prefix = f"{Colors.RED}{Colors.BOLD}*" if idx in top_3_indices else " "
                suffix = f"{Colors.ENDC}" if idx in top_3_indices else ""
                print(f"{prefix} Sent {idx:<2} | Score: {total:+.4f} | {text_preview}{suffix}")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved to: {json_output_path}")


if __name__ == "__main__":
    main()