import json
import os
import argparse
import html
import math

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Agent Attribution Analysis</title>
    <style>
        :root {{
            --traj-base-hue: 210;
            --sent-base-hue: 0;
        }}
        body {{
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        .header {{ border-bottom: 2px solid #eaeaea; padding-bottom: 20px; margin-bottom: 30px; }}
        .header h1 {{ margin: 0; color: #2c3e50; font-size: 24px; }}
        .meta-box {{
            background: #f8f9fa; border-left: 4px solid #3498db;
            padding: 15px; margin-top: 15px; border-radius: 4px; font-size: 14px;
        }}
        .target-response-box {{
            margin-top: 10px; border-top: 1px solid #ddd; padding-top: 10px;
            font-family: monospace; color: #444; background: #fff; padding: 10px;
            border: 1px solid #eee; border-radius: 4px; white-space: pre-wrap;
            max-height: 300px; overflow-y: auto;
        }}
        .summary-section {{ margin-bottom: 40px; overflow-x: auto; }}
        .summary-title {{ font-size: 18px; font-weight: 600; margin-bottom: 10px; color: #2c3e50; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f1f2f6; font-weight: 600; color: #555; }}
        tr:hover {{ background-color: #f9f9f9; }}
        .score-cell {{ font-family: monospace; font-weight: bold; }}
        .top-k-row {{ background-color: #fff3e0 !important; border-left: 3px solid #e67e22; }}
        .sent-preview {{ display: block; margin-bottom: 4px; color: #666; font-style: italic; }}
        .sent-score-tag {{ font-size: 0.85em; background: #ffebee; color: #c0392b; padding: 1px 4px; border-radius: 3px; margin-right: 5px; }}
        .rank-badge-table {{
            display: inline-block; background: #e67e22; color: white;
            padding: 1px 6px; border-radius: 10px; font-size: 11px; font-weight: bold;
            margin-right: 5px;
        }}
        .legend {{
            margin-bottom: 20px; padding: 10px; background: #fff;
            border: 1px dashed #ccc; font-size: 0.9em; border-radius: 4px;
            display: flex; gap: 20px; align-items: center;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .color-box {{ width: 20px; height: 20px; border-radius: 3px; border: 1px solid #ccc; }}
        .message-row {{ display: flex; margin-bottom: 25px; flex-direction: column; }}
        .msg-header {{
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 6px; font-size: 13px; font-weight: 600; color: #555; padding: 0 5px;
        }}
        .role-badge {{
            text-transform: uppercase; padding: 2px 8px; border-radius: 4px; font-size: 11px; color: white;
        }}
        .role-user {{ background-color: #3498db; }}
        .role-assistant {{ background-color: #2ecc71; }}
        .role-system {{ background-color: #95a5a6; }}
        .role-tool {{ background-color: #e67e22; }}
        .msg-content {{
            padding: 15px; border-radius: 8px; border: 1px solid #e1e4e8;
            position: relative; white-space: pre-wrap; word-wrap: break-word;
            transition: all 0.2s;
            background-color: #fff;
        }}
        .top-k-highlight .msg-content {{
            border: 2px solid #e74c3c;
            box-shadow: 0 0 10px rgba(231, 76, 60, 0.2);
            background-color: #fff !important;
        }}
        .top-k-label {{
            color: #e74c3c; font-weight: bold; font-size: 12px; margin-left: 10px;
            display: inline-block; background: #ffebee; padding: 2px 8px; border-radius: 4px;
            border: 1px solid #ffcdd2;
        }}
        .sentence-span {{
            border-radius: 3px; cursor: help;
        }}
        .sentence-span:hover {{ outline: 1px solid #333; z-index: 5; position: relative; }}
        .tooltip {{ position: relative; }}
        .tooltip .tooltiptext {{
            visibility: hidden; width: 240px; background-color: rgba(0, 0, 0, 0.9);
            color: #fff; text-align: left; border-radius: 6px; padding: 8px 12px;
            position: absolute; z-index: 10; bottom: 130%; left: 50%; margin-left: -120px;
            opacity: 0; transition: opacity 0.2s; font-size: 12px; line-height: 1.4;
            pointer-events: none; box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
        .tooltip .tooltiptext::after {{
            content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px;
            border-width: 5px; border-style: solid; border-color: rgba(0,0,0,0.9) transparent transparent transparent;
        }}
        .tooltip:hover .tooltiptext {{ visibility: visible; opacity: 1; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Agent Trajectory Attribution Map</h1>
            <div class="meta-box">
                <div><strong>Source File:</strong> {source_file}</div>
                <div><strong>Model:</strong> {model_name}</div>
                <div style="margin-top:8px;">
                    <strong>Target Response (Full):</strong>
                    <div class="target-response-box">{target_response}</div>
                </div>
            </div>
        </div>
        <div class="summary-section">
            <div class="summary-title">Attribution Summary</div>
            <table>
                <thead>
                    <tr>
                        <th style="width: 60px;">Step</th>
                        <th style="width: 80px;">Role</th>
                        <th style="width: 100px;">Traj Score<br><small>(Prob Diff)</small></th>
                        <th>Top Contributing Sentences (for Top-K steps)</th>
                    </tr>
                </thead>
                <tbody>
                    {summary_rows}
                </tbody>
            </table>
        </div>
        <div class="legend">
            <strong>Legend:</strong>
            <div class="legend-item">
                <div class="color-box" style="background-color: rgba(52, 152, 219, 0.3)"></div>
                <span>Trajectory Score (Blue Background)</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="border: 2px solid #e74c3c;"></div>
                <span>Top-K Impact Step</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background-color: rgba(231, 76, 60, 0.5)"></div>
                <span>High Impact Sentence (Red Heatmap)</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background-color: #fff; border: 1px solid #ccc;"></div>
                <span>No Attribution Data</span>
            </div>
        </div>
        <div class="chat-flow">
            {chat_content}
        </div>
    </div>
</body>
</html>
"""

def get_traj_bg_color(score, min_val, max_val):
    if max_val == min_val:
        norm = 0.5
    else:
        norm = (score - min_val) / (max_val - min_val)
    norm = max(0.0, min(1.0, norm))
    alpha = 0.05 + (norm * 0.35)
    return f"rgba(52, 152, 219, {alpha:.3f})"

def get_sent_bg_color(score, min_val, max_val):
    if max_val == min_val:
        norm = 0.5
    else:
        norm = (score - min_val) / (max_val - min_val)
    norm = max(0.0, min(1.0, norm))
    alpha = 0.05 + (norm * 0.75)
    return f"rgba(231, 76, 60, {alpha:.3f})"

def escape_text(text):
    return html.escape(text)

def truncate_text(text, limit=80):
    text = text.replace('\n', ' ')
    if len(text) > limit:
        return text[:limit] + "..."
    return text

def main():
    parser = argparse.ArgumentParser(description="Visualize Attribution Full Flow")
    parser.add_argument("--traj_attr_file", type=str, required=True)
    parser.add_argument("--sent_attr_file", type=str, required=True)
    parser.add_argument("--original_traj_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="full_attribution_vis.html")
    args = parser.parse_args()

    with open(args.traj_attr_file, 'r') as f: traj_attr_data = json.load(f)
    with open(args.sent_attr_file, 'r') as f: sent_attr_data = json.load(f)
    with open(args.original_traj_file, 'r') as f: original_data = json.load(f)

    full_trajectory = original_data.get('trajectory') or original_data.get('trace')

    sent_map = {item['step_index']: item for item in sent_attr_data['sentence_attribution']}
    traj_attr_map = {item['step_index']: item for item in traj_attr_data['trajectory_analysis']}

    top_k_indices = [item['step_index'] for item in sent_attr_data['sentence_attribution']]
    top_k_indices.sort(key=lambda idx: traj_attr_map.get(idx, {}).get('metrics', {}).get('llr_score', -9999), reverse=True)
    rank_map = {idx: i+1 for i, idx in enumerate(top_k_indices)}

    target_response_text = "..."
    if full_trajectory and len(full_trajectory) > 0:
        target_response_text = full_trajectory[-1]['content']
    else:
        target_response_text = traj_attr_data.get('meta_info', {}).get('target_response_preview', '...')

    step_scores = [x['metrics']['llr_score'] for x in traj_attr_data['trajectory_analysis'] if x['step_index'] > 0]
    sent_scores = []
    for item in sent_attr_data['sentence_attribution']:
        for sent in item['sentence_analysis']:
            sent_scores.append(sent['scores']['total_score'])

    if not step_scores: step_scores = [0]
    if not sent_scores: sent_scores = [0]

    max_step = max(step_scores)
    min_step = min(step_scores)
    max_sent = max(sent_scores)
    min_sent = min(sent_scores)

    summary_rows_html = ""
    for i, turn in enumerate(full_trajectory):
        attr_idx = i + 1
        role = turn['role']
        step_item = traj_attr_map.get(attr_idx)
        llr_score = step_item['metrics']['llr_score'] if step_item else 0.0
        rank = rank_map.get(attr_idx)
        is_top_k = rank is not None
        row_class = "top-k-row" if is_top_k else ""
        rank_html = f"<span class='rank-badge-table'>Top {rank}</span>" if is_top_k else ""
        sent_details_html = ""
        if is_top_k and attr_idx in sent_map:
            sent_data = sent_map[attr_idx]
            sorted_sents = sorted(sent_data['sentence_analysis'], key=lambda x: x['scores']['total_score'], reverse=True)
            for s in sorted_sents[:3]:
                sc = s['scores']['total_score']
                txt = truncate_text(s['text'], 60)
                sent_details_html += f"<div class='sent-preview'><span class='sent-score-tag'>{sc:+.3f}</span> {escape_text(txt)}</div>"
        else:
            sent_details_html = "<span style='color:#999;'>-</span>" if step_item else "<span style='color:#ccc; font-size:0.9em'>(No Attribution)</span>"
        score_display = f"{llr_score:+.4f}" if step_item else "<span style='color:#ccc'>N/A</span>"
        summary_rows_html += f"""
        <tr class="{row_class}">
            <td>{rank_html}{i}</td>
            <td><span class="role-badge role-{role}">{role}</span></td>
            <td class="score-cell">{score_display}</td>
            <td>{sent_details_html}</td>
        </tr>
        """

    chat_content_html = ""
    for i, turn in enumerate(full_trajectory):
        attr_idx = i + 1
        role = turn['role']
        content = turn['content']
        step_item = traj_attr_map.get(attr_idx)
        if step_item:
            llr_score = step_item['metrics'].get('llr_score', 0)
            header_score_display = f"Diff: {llr_score:+.4f}"
            header_tooltip = f"Traj Index: {i}<br>Attr Index: {attr_idx}<br>Prob Diff: {llr_score:.4f}<br>LLR: {llr_score:.4f}"
            tooltip_cls = "tooltip"
            cursor_style = "cursor:help;"
        else:
            header_score_display = "<span style='color:#aaa'>N/A</span>"
            header_tooltip = "This step was not included in attribution analysis."
            tooltip_cls = "tooltip"
            cursor_style = "cursor:default; color: #aaa;"
        rank = rank_map.get(attr_idx)
        is_top_k = rank is not None
        sent_data = sent_map.get(attr_idx)
        container_class = "message-row" + (" top-k-highlight" if is_top_k else "")
        top_k_label = f"<span class='top-k-label'>★ Top {rank} Impact</span>" if is_top_k else ""
        role_class = f"role-{role}" if role in ['user', 'assistant', 'system', 'tool'] else "role-user"
        content_inner_html = ""
        content_style = ""
        if is_top_k and sent_data:
            for s_info in sent_data['sentence_analysis']:
                bg_col = get_sent_bg_color(s_info['scores']['total_score'], min_sent, max_sent)
                tooltip_txt = f"<strong>Sentence Score: {s_info['scores']['total_score']:.4f}</strong><br>Drop: {s_info['scores']['drop_score']:.4f}<br>Hold: {s_info['scores']['hold_score']:.4f}"
                content_inner_html += f"<span class=\"sentence-span tooltip\" style=\"background-color: {bg_col}\">{escape_text(s_info['text'])}<span class=\"tooltiptext\">{tooltip_txt}</span></span>"
        else:
            if step_item:
                content_style = f"style='background-color: {get_traj_bg_color(llr_score, min_step, max_step)}'"
            else:
                content_style = "style='background-color: #fff; border: 1px dashed #ddd; color: #777;'"
            content_inner_html = escape_text(content)
        chat_content_html += f"""
        <div class="{container_class}">
            <div class="msg-header">
                <div>
                    <span class="role-badge {role_class}">{role}</span>
                    <span style="margin-left: 8px;">Step {i}</span>
                    {top_k_label}
                </div>
                <div class="{tooltip_cls}" style="{cursor_style}">
                    <span style="font-family:monospace; background:#eee; padding:2px 5px; border-radius:3px;">
                        {header_score_display}
                    </span>
                    <span class="tooltiptext">{header_tooltip}</span>
                </div>
            </div>
            <div class="msg-content" {content_style}>
                {content_inner_html}
            </div>
        </div>
        """

    meta = traj_attr_data.get('meta_info', {})
    final_html = HTML_TEMPLATE.format(
        source_file=meta.get('source_file', 'unknown'),
        model_name=meta.get('model', 'unknown'),
        target_response=escape_text(target_response_text),
        summary_rows=summary_rows_html,
        chat_content=chat_content_html
    )

    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(final_html)

if __name__ == "__main__":
    main()