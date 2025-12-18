#!/usr/bin/env python3
"""Quick comparison script for evaluation results."""

import json
import sys

if len(sys.argv) < 3:
    print("Usage: python compare_results.py <original_eval.json> <experimental_eval.json>")
    sys.exit(1)

# Load both evaluation results
with open(sys.argv[1], 'r') as f:
    orig = json.load(f)
with open(sys.argv[2], 'r') as f:
    exp = json.load(f)

print('=' * 80)
print('COMPARISON: Original vs Experimental Pipeline')
print('=' * 80)
print()

# Answer Quality
print('📊 ANSWER QUALITY:')
print('-' * 80)
print(f"{'Metric':<20} {'Original':<30} {'Experimental':<30} {'Difference':<15}")
print('-' * 80)

orig_em = orig['answer_metrics']['em']
exp_em = exp['answer_metrics']['em']
orig_em_ci = orig['answer_metrics'].get('ci_95_em', {})
exp_em_ci = exp['answer_metrics'].get('ci_95_em', {})
if isinstance(orig_em_ci, dict):
    orig_em_str = f"{orig_em:.4f} [{orig_em_ci.get('lower', 'N/A'):.4f}, {orig_em_ci.get('upper', 'N/A'):.4f}]"
else:
    orig_em_str = f"{orig_em:.4f}"
if isinstance(exp_em_ci, dict):
    exp_em_str = f"{exp_em:.4f} [{exp_em_ci.get('lower', 'N/A'):.4f}, {exp_em_ci.get('upper', 'N/A'):.4f}]"
else:
    exp_em_str = f"{exp_em:.4f}"
print(f"{'EM':<20} {orig_em_str:<30} {exp_em_str:<30} {exp_em - orig_em:+.4f}")

orig_f1 = orig['answer_metrics']['f1']
exp_f1 = exp['answer_metrics']['f1']
orig_f1_ci = orig['answer_metrics'].get('ci_95_f1', {})
exp_f1_ci = exp['answer_metrics'].get('ci_95_f1', {})
if isinstance(orig_f1_ci, dict):
    orig_f1_str = f"{orig_f1:.4f} [{orig_f1_ci.get('lower', 'N/A'):.4f}, {orig_f1_ci.get('upper', 'N/A'):.4f}]"
else:
    orig_f1_str = f"{orig_f1:.4f}"
if isinstance(exp_f1_ci, dict):
    exp_f1_str = f"{exp_f1:.4f} [{exp_f1_ci.get('lower', 'N/A'):.4f}, {exp_f1_ci.get('upper', 'N/A'):.4f}]"
else:
    exp_f1_str = f"{exp_f1:.4f}"
print(f"{'F1':<20} {orig_f1_str:<30} {exp_f1_str:<30} {exp_f1 - orig_f1:+.4f}")
print()

# Supporting Facts
print('📊 SUPPORTING FACTS QUALITY:')
print('-' * 80)
orig_sp_f1 = orig['supporting_facts_metrics']['sp_f1']
exp_sp_f1 = exp['supporting_facts_metrics']['sp_f1']
orig_sp_f1_ci = orig['supporting_facts_metrics'].get('ci_95_f1', {})
exp_sp_f1_ci = exp['supporting_facts_metrics'].get('ci_95_f1', {})
if isinstance(orig_sp_f1_ci, dict):
    orig_sp_f1_str = f"{orig_sp_f1:.4f} [{orig_sp_f1_ci.get('lower', 'N/A'):.4f}, {orig_sp_f1_ci.get('upper', 'N/A'):.4f}]"
else:
    orig_sp_f1_str = f"{orig_sp_f1:.4f}"
if isinstance(exp_sp_f1_ci, dict):
    exp_sp_f1_str = f"{exp_sp_f1:.4f} [{exp_sp_f1_ci.get('lower', 'N/A'):.4f}, {exp_sp_f1_ci.get('upper', 'N/A'):.4f}]"
else:
    exp_sp_f1_str = f"{exp_sp_f1:.4f}"
print(f"{'F1':<20} {orig_sp_f1_str:<30} {exp_sp_f1_str:<30} {exp_sp_f1 - orig_sp_f1:+.4f}")

orig_sp_recall = orig['supporting_facts_metrics']['sp_recall']
exp_sp_recall = exp['supporting_facts_metrics']['sp_recall']
print(f"{'Recall':<20} {orig_sp_recall:.4f} {'':<28} {exp_sp_recall:.4f} {'':<28} {exp_sp_recall - orig_sp_recall:+.4f}")
print()

# Token Efficiency
print('📊 TOKEN EFFICIENCY:')
print('-' * 80)
orig_tokens = orig['token_efficiency']['avg_tokens']
exp_tokens = exp['token_efficiency']['avg_tokens']
print(f"{'Avg Tokens/Q':<20} {orig_tokens:.1f} {'':<28} {exp_tokens:.1f} {'':<28} {exp_tokens - orig_tokens:+.1f}")

orig_acc_per_1k = orig['token_efficiency']['accuracy_per_1k_tokens']
exp_acc_per_1k = exp['token_efficiency']['accuracy_per_1k_tokens']
print(f"{'Acc per 1k tokens':<20} {orig_acc_per_1k:.4f} {'':<28} {exp_acc_per_1k:.4f} {'':<28} {exp_acc_per_1k - orig_acc_per_1k:+.4f}")
print()

# Summary
print('=' * 80)
print('📈 SUMMARY:')
print('=' * 80)
if exp_em > orig_em:
    print(f'✅ Experimental pipeline has HIGHER EM: {exp_em:.4f} vs {orig_em:.4f} (+{exp_em - orig_em:.4f})')
else:
    print(f'❌ Experimental pipeline has LOWER EM: {exp_em:.4f} vs {orig_em:.4f} ({exp_em - orig_em:.4f})')

if exp_f1 > orig_f1:
    print(f'✅ Experimental pipeline has HIGHER F1: {exp_f1:.4f} vs {orig_f1:.4f} (+{exp_f1 - orig_f1:.4f})')
else:
    print(f'❌ Experimental pipeline has LOWER F1: {exp_f1:.4f} vs {orig_f1:.4f} ({exp_f1 - orig_f1:.4f})')

if exp_tokens > orig_tokens:
    print(f'⚠️  Experimental uses MORE tokens: {exp_tokens:.1f} vs {orig_tokens:.1f} (+{exp_tokens - orig_tokens:.1f})')
else:
    print(f'✅ Experimental uses FEWER tokens: {exp_tokens:.1f} vs {orig_tokens:.1f} ({exp_tokens - orig_tokens:.1f})')
print()
total_q = orig.get('summary', {}).get('total_questions', orig.get('supporting_facts_metrics', {}).get('total', 'N/A'))
print(f'Total questions tested: {total_q}')
print('=' * 80)
