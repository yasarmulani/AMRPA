# =============================================================================
# AMRPA ABLATION STUDY - HotpotQA
# =============================================================================
# HOW TO USE ON KAGGLE:
#   Change VARIANT below to one of:
#     "full"           → Full AMRPA (all 4 claims active)
#     "no_gate"        → Ablate Claim 1 (G = 1 always)
#     "uniform_memory" → Ablate Claim 2 (uniform alpha, no MLP)
#     "no_decay"       → Ablate Claim 3 (gamma = 1.0)
#     "fixed_depth"    → Ablate Claim 4 (window always = 1)
#   Run all → Save as separate Kaggle version for each variant
# =============================================================================

print("Starting AMRPA Ablation...")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
import string
import re
import math
import gc
import time
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# ★  CHANGE THIS ONE LINE BETWEEN KAGGLE RUNS  ★
# ==============================================================================
VARIANT = "full"
# "full" | "no_gate" | "uniform_memory" | "no_decay" | "fixed_depth"
# ==============================================================================

VARIANT_DESCRIPTIONS = {
    "full":           "Full AMRPA (all 4 claims active)",
    "no_gate":        "-Claim 1: No Smart Gate (G = 1.0 always)",
    "uniform_memory": "-Claim 2: No Dynamic Selection (uniform alpha)",
    "no_decay":       "-Claim 3: No Fading Ink (gamma = 1.0)",
    "fixed_depth":    "-Claim 4: No Adaptive Depth (fixed window = 1)",
}
assert VARIANT in VARIANT_DESCRIPTIONS, f"Unknown VARIANT: {VARIANT}"

print("=" * 80)
print(f"AMRPA ABLATION STUDY — HotpotQA")
print(f"VARIANT     : {VARIANT}")
print(f"DESCRIPTION : {VARIANT_DESCRIPTIONS[VARIANT]}")
print("=" * 80)


# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    PROCESSED_DATA_DIR = '/kaggle/input/tokenized-hotpot'
    BATCH_SIZE = 32
    MAX_SEQ_LEN = 384
    N_AMRPA_LAYERS = 4
    N_HEADS = 4
    D_MODEL = 768
    D_K = 192
    D_MLP = 384
    GAMMA = 0.9
    EPSILON = 0.001
    ALPHA_TEMPERATURE = 0.25
    EPOCHS = 10
    LR = 2e-5
    WEIGHT_DECAY = 0.005
    LABEL_SMOOTHING = 0.05
    GRADIENT_ACCUMULATION_STEPS = 1
    WARMUP_RATIO = 0.1
    DROPOUT = 0.2
    DIVERSITY_WEIGHT = 0.005
    GATE_REG_WEIGHT = 0.05
    PATIENCE = 8

config = Config()

# Effective gamma: 1.0 when ablating Claim 3, else 0.9
EFFECTIVE_GAMMA = 1.0 if VARIANT == "no_decay" else config.GAMMA

print(f"\n[Config] Effective gamma = {EFFECTIVE_GAMMA}")
print(f"[Config] Gate mode       = {'DISABLED (G=1)' if VARIANT == 'no_gate' else 'ACTIVE (sigmoid)'}")
print(f"[Config] Alpha mode      = {'UNIFORM (1/w)' if VARIANT == 'uniform_memory' else 'MLP-learned'}")
print(f"[Config] Window mode     = {'FIXED=1' if VARIANT == 'fixed_depth' else 'ADAPTIVE (log2)'}\n")


# ==============================================================================
# AMRPA SELF-ATTENTION WRAPPER
# All 4 ablation switches live inside this class
# ==============================================================================
class AMRPA_SelfAttentionWrapper(nn.Module):

    def __init__(self, original_attention, layer_idx):
        super().__init__()
        self.original = original_attention
        self.layer_idx = layer_idx
        self.attention_history = None
        self.last_metrics = {}

        self.d_model = original_attention.query.in_features
        self.d_k = self.d_model

        # Claim 1 params (learnable gate)
        self.gamma_g = nn.Parameter(torch.tensor(2.0))
        self.bias_g  = nn.Parameter(torch.tensor(-0.25))

        # Claim 2 params (MLP alpha selector)
        # Always initialised so checkpoint shapes stay consistent
        self.mlp_alpha = nn.Sequential(
            nn.Linear(self.d_k * 2, config.D_MLP),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.D_MLP, 1),
        )

        self.w_mem          = nn.Linear(self.d_k, self.d_k, bias=False)
        self.proj_attention = nn.Linear(self.d_k, self.d_k, bias=False)

    # ------------------------------------------------------------------
    # CLAIM 4: Adaptive window size
    # ------------------------------------------------------------------
    def adaptive_window_size(self, l):
        if VARIANT == "fixed_depth":
            return 1                          # Claim 4 ABLATED: fixed window
        if l <= 2:
            return 1
        elif 2 < l <= 8:
            return math.floor(math.log2(l)) + 1
        else:
            return 4

    # ------------------------------------------------------------------
    # Core memory computation
    # ------------------------------------------------------------------
    def compute_memory_bias(self, Q, K, V, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        gate_impact_ps         = torch.zeros(batch_size, device=device)
        alpha_diversity_ps     = torch.zeros(batch_size, device=device)
        memory_contribution_ps = torch.zeros(batch_size, device=device)
        gate_variance_ps       = torch.zeros(batch_size, device=device)

        total_roberta_layers = 12
        amrpa_start_layer    = total_roberta_layers - config.N_AMRPA_LAYERS + 1
        relative_layer_idx   = self.layer_idx - amrpa_start_layer + 1

        if (relative_layer_idx > 1
                and self.attention_history is not None
                and len(self.attention_history) > 0):

            w = self.adaptive_window_size(relative_layer_idx)
            memory_window = self.attention_history[-min(relative_layer_idx - 1, w):]

            alpha_scores     = []
            decayed_patterns = []

            for k, past_attention in enumerate(reversed(memory_window), 1):

                # CLAIM 3: Fading Ink
                # FULL:     decay_factor = GAMMA^k  (older layers matter less)
                # NO_DECAY: decay_factor = 1.0      (all history equally weighted)
                decay_factor = EFFECTIVE_GAMMA ** k       # Claim 3 switch

                noise     = torch.rand_like(past_attention) * config.EPSILON
                decayed_A = decay_factor * past_attention + noise
                decayed_patterns.append(decayed_A)

                # CLAIM 2: Dynamic Memory Selection
                # FULL:           compute MLP alpha score for this memory slot
                # UNIFORM_MEMORY: skip MLP, use uniform weights after loop
                if VARIANT != "uniform_memory":           # Claim 2 switch
                    projected_values = torch.matmul(decayed_A, V)
                    proj_A           = self.proj_attention(projected_values)
                    alpha_input      = torch.cat([Q, proj_A], dim=-1)
                    alpha_score      = self.mlp_alpha(alpha_input)
                    alpha_scores.append(alpha_score)

            if decayed_patterns:

                # Build alpha_weights
                if VARIANT == "uniform_memory":
                    # Claim 2 ABLATED: every memory slot gets equal weight 1/w
                    w_actual      = len(decayed_patterns)
                    alpha_weights = torch.full(
                        (batch_size, seq_len, w_actual),
                        fill_value=1.0 / w_actual,
                        device=device,
                    )
                else:
                    alpha_tensor  = torch.cat(alpha_scores, dim=-1)   # (B, S, w)
                    alpha_weights = F.softmax(
                        alpha_tensor / config.ALPHA_TEMPERATURE, dim=-1
                    )

                # Alpha diversity (entropy) per sample
                token_entropy      = -(alpha_weights * torch.log(alpha_weights + 1e-9)).sum(dim=-1)
                alpha_diversity_ps = token_entropy.mean(dim=1)         # (B,)

                # Weighted memory blend
                memory_stack = torch.stack(decayed_patterns, dim=-1)  # (B, S, S, w)
                M = torch.sum(alpha_weights.unsqueeze(2) * memory_stack, dim=-1)

                MV     = torch.matmul(M, V)
                M_proj = self.proj_attention(MV)
                sim_score = (Q * M_proj).sum(dim=-1) / math.sqrt(self.d_k)

                # CLAIM 1: Smart Gatekeeper
                # FULL:    G = sigmoid(gamma_g * sim + bias_g)  learnable gate
                # NO_GATE: G = 1.0                              no filtering
                if VARIANT == "no_gate":
                    G = torch.ones_like(sim_score)            # Claim 1 ABLATED
                else:
                    G = torch.sigmoid(self.gamma_g * sim_score + self.bias_g)

                gate_impact_ps   = G.mean(dim=1)              # (B,)
                gate_variance_ps = G.var(dim=1)               # (B,)

                M_transformed   = self.w_mem(torch.matmul(M, V))
                gated_memory    = G.unsqueeze(-1) * M_transformed

                token_norms            = gated_memory.norm(dim=-1)
                memory_contribution_ps = token_norms.mean(dim=1)

                memory_bias  = torch.matmul(gated_memory, K.transpose(-2, -1)) / math.sqrt(self.d_k)

                self.last_metrics = {
                    'gate_impact':         gate_impact_ps.detach().cpu(),
                    'alpha_diversity':     alpha_diversity_ps.detach().cpu(),
                    'memory_contribution': memory_contribution_ps.detach().cpu(),
                    'gate_variance':       gate_variance_ps.detach().cpu(),
                    'using_memory':        torch.ones(batch_size, dtype=torch.float32),
                }
                return memory_bias

        # No memory case
        self.last_metrics = {
            'gate_impact':         torch.zeros(batch_size, dtype=torch.float32),
            'alpha_diversity':     torch.zeros(batch_size, dtype=torch.float32),
            'memory_contribution': torch.zeros(batch_size, dtype=torch.float32),
            'gate_variance':       torch.zeros(batch_size, dtype=torch.float32),
            'using_memory':        torch.zeros(batch_size, dtype=torch.float32),
        }
        return torch.zeros_like(torch.matmul(Q, K.transpose(-2, -1)))

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, past_key_values=None,
                output_attentions=False, **kwargs):

        Q = self.original.query(hidden_states)
        K = self.original.key(hidden_states)
        V = self.original.value(hidden_states)

        base_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attention_mask = attention_mask.squeeze(1)
            if attention_mask.dim() == 3 and attention_mask.size(1) == 1:
                attention_mask = attention_mask.expand(-1, base_scores.size(1), -1)
            base_scores = base_scores + attention_mask

        memory_bias  = self.compute_memory_bias(Q, K, V, hidden_states)
        final_scores = base_scores + memory_bias

        attention_probs = F.softmax(final_scores, dim=-1)
        context_layer   = torch.matmul(attention_probs, V)

        if self.attention_history is not None:
            self.attention_history.append(attention_probs.detach())

        return (context_layer, attention_probs) if output_attentions else (context_layer,)

    def reset_metrics(self):
        self.last_metrics = {}


# ==============================================================================
# MODEL
# ==============================================================================
class RoBERTa_AMRPA_QA(nn.Module):
    def __init__(self, n_amrpa_layers=4):
        super().__init__()
        print(f"\n{'='*80}")
        print(f"MODEL: RoBERTa + AMRPA  |  VARIANT: {VARIANT}")
        print(f"{'='*80}")

        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.d_model = self.roberta.config.hidden_size

        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False

        total_layers = len(self.roberta.encoder.layer)
        freeze_until = max(0, total_layers - 8)
        for i, layer in enumerate(self.roberta.encoder.layer):
            for param in layer.parameters():
                param.requires_grad = (i >= freeze_until)

        print(f"  Layers 0-{freeze_until-1} frozen | Layers {freeze_until}-{total_layers-1} trainable")

        start_layer         = total_layers - n_amrpa_layers
        self.amrpa_wrappers = []

        for i in range(start_layer, total_layers):
            layer_idx  = i + 1
            orig_attn  = self.roberta.encoder.layer[i].attention.self
            wrapper    = AMRPA_SelfAttentionWrapper(orig_attn, layer_idx)
            self.roberta.encoder.layer[i].attention.self = wrapper
            self.amrpa_wrappers.append(wrapper)
            print(f"  ✓ Layer {i} → AMRPA wrapper (layer_idx={layer_idx})")

        self.qa_outputs = nn.Linear(self.d_model, 2)

    def forward(self, input_ids, attention_mask, return_metrics=False):
        current_history = []
        for wrapper in self.amrpa_wrappers:
            wrapper.attention_history = current_history
            wrapper.reset_metrics()

        out          = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        seq_out      = out.last_hidden_state
        logits       = self.qa_outputs(seq_out)
        start_logits = logits[:, :, 0]
        end_logits   = logits[:, :, 1]

        if return_metrics:
            all_metrics = []
            for wrapper in self.amrpa_wrappers:
                if hasattr(wrapper, 'last_metrics') and wrapper.last_metrics:
                    all_metrics.append(wrapper.last_metrics)
                else:
                    bs = start_logits.size(0)
                    all_metrics.append({
                        k: torch.zeros(bs, dtype=torch.float32)
                        for k in ['gate_impact', 'alpha_diversity',
                                  'memory_contribution', 'gate_variance', 'using_memory']
                    })
            return start_logits, end_logits, all_metrics

        return start_logits, end_logits


# ==============================================================================
# DATASET
# ==============================================================================
class PreprocessedQADataset(torch.utils.data.Dataset):
    def __init__(self, path):
        print(f"Loading: {path}")
        self.data = torch.load(path)
        print(f"  ✓ {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids':       torch.tensor(item['input_ids'],       dtype=torch.long),
            'attention_mask':  torch.tensor(item['attention_mask'],  dtype=torch.long),
            'start_positions': torch.tensor(item['start_positions'], dtype=torch.long),
            'end_positions':   torch.tensor(item['end_positions'],   dtype=torch.long),
            'answer_text':     item.get('answer_text', ""),
        }


# ==============================================================================
# EVALUATION HELPERS
# ==============================================================================
def normalize_answer(s):
    def remove_articles(t): return re.sub(r'\b(a|an|the)\b', ' ', t)
    def white_space_fix(t):  return ' '.join(t.split())
    def remove_punc(t):      return ''.join(ch for ch in t if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_exact_match(pred, truth):
    return int(normalize_answer(pred) == normalize_answer(truth))

def compute_f1(pred, truth):
    p_tok = normalize_answer(pred).split()
    t_tok = normalize_answer(truth).split()
    if not p_tok or not t_tok:
        return int(p_tok == t_tok)
    common = set(p_tok) & set(t_tok)
    if not common:
        return 0
    prec = len(common) / len(p_tok)
    rec  = len(common) / len(t_tok)
    return 2 * prec * rec / (prec + rec)

def get_best_span(start_logits, end_logits, max_len=30):
    s = torch.argmax(start_logits).item()
    e = torch.argmax(end_logits).item()
    if e < s:   e = s
    if e - s + 1 > max_len: e = s + max_len - 1
    return s, e


# ==============================================================================
# TRAINING LOOP
# ==============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss  = 0.0
    all_metrics = defaultdict(list)
    loss_fct    = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=config.LABEL_SMOOTHING)
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [TRAIN]")
    for batch_idx, batch in enumerate(pbar):
        input_ids       = batch['input_ids'].to(device)
        attention_mask  = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions   = batch['end_positions'].to(device)

        start_logits, end_logits, metrics_list = model(
            input_ids, attention_mask, return_metrics=True
        )

        qa_loss = (loss_fct(start_logits, start_positions) +
                   loss_fct(end_logits,   end_positions)) / 2.0

        diversity_loss      = 0.0
        gate_regularization = 0.0
        if metrics_list:
            for lm in metrics_list:
                a_div  = lm.get('alpha_diversity', torch.zeros(1)).mean().item()
                g_mean = lm.get('gate_impact',      torch.zeros(1)).mean().item()
                if a_div < 0.15:
                    diversity_loss += (0.15 - a_div) ** 2
                if g_mean < 0.1:
                    gate_regularization += (0.1 - g_mean) ** 2
                elif g_mean > 0.95:
                    gate_regularization += (g_mean - 0.95) ** 2
            diversity_loss      /= len(metrics_list)
            gate_regularization /= len(metrics_list)

        loss = (qa_loss
                + config.DIVERSITY_WEIGHT * diversity_loss
                + config.GATE_REG_WEIGHT  * gate_regularization)
        (loss / config.GRADIENT_ACCUMULATION_STEPS).backward()

        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        for m in metrics_list:
            for k, v in m.items():
                all_metrics[k].extend(
                    v.tolist() if isinstance(v, torch.Tensor) else [float(v)]
                )

        pf = {'loss': f'{loss.item():.4f}'}
        if all_metrics.get('gate_impact'):
            pf['gate']  = f'{np.mean(all_metrics["gate_impact"][-32:]):.3f}'
        if all_metrics.get('memory_contribution'):
            pf['mem']   = f'{np.mean(all_metrics["memory_contribution"][-32:]):.3f}'
        if all_metrics.get('alpha_diversity'):
            pf['a_div'] = f'{np.mean(all_metrics["alpha_diversity"][-32:]):.3f}'
        pbar.set_postfix(pf)

        # First batch mechanism check
        if batch_idx == 0 and epoch == 0:
            print(f"\n{'='*60}")
            print(f"MECHANISM CHECK — First Batch | VARIANT: {VARIANT}")
            print(f"{'='*60}")
            for li, lm in enumerate(metrics_list, 1):
                print(f"  AMRPA Layer {li}:")
                for k, v in lm.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k:25s}: {v.mean().item():.4f}  (±{v.std().item():.4f})")
                    else:
                        print(f"    {k:25s}: {float(v):.4f}")
            print(f"{'='*60}\n")

        if batch_idx % 20 == 0:
            torch.cuda.empty_cache(); gc.collect()

    avg_loss    = total_loss / len(dataloader)
    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}

    print(f"\n{'─'*60}")
    print(f"Train Epoch Summary [{VARIANT}]:")
    for k in ['gate_impact', 'gate_variance', 'alpha_diversity', 'memory_contribution']:
        print(f"  {k:25s}: {avg_metrics.get(k, 0.0):.4f}")
    print(f"{'─'*60}")

    return avg_loss, avg_metrics


# ==============================================================================
# EVALUATION LOOP
# ==============================================================================
def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    total_loss  = 0.0
    all_em      = []
    all_f1      = []
    all_metrics = defaultdict(list)
    loss_fct    = nn.CrossEntropyLoss(ignore_index=-1)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            input_ids       = batch['input_ids'].to(device)
            attention_mask  = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions   = batch['end_positions'].to(device)
            answer_texts    = batch['answer_text']

            start_logits, end_logits, metrics_list = model(
                input_ids, attention_mask, return_metrics=True
            )
            loss = (loss_fct(start_logits, start_positions) +
                    loss_fct(end_logits,   end_positions)) / 2.0
            total_loss += loss.item()

            for m in metrics_list:
                for k, v in m.items():
                    all_metrics[k].extend(
                        v.tolist() if isinstance(v, torch.Tensor) else [float(v)]
                    )

            for i in range(input_ids.size(0)):
                s, e  = get_best_span(start_logits[i], end_logits[i])
                pred  = tokenizer.decode(input_ids[i][s:e+1], skip_special_tokens=True)
                truth = answer_texts[i]
                all_em.append(compute_exact_match(pred, truth))
                all_f1.append(compute_f1(pred, truth))

            pf = {
                'loss': f'{loss.item():.4f}',
                'em':   f'{np.mean(all_em):.3f}'  if all_em else '0.000',
                'f1':   f'{np.mean(all_f1):.3f}'  if all_f1 else '0.000',
            }
            if all_metrics.get('gate_impact'):
                pf['gate'] = f'{np.mean(all_metrics["gate_impact"]):.3f}'
            pbar.set_postfix(pf)

    avg_loss    = total_loss / len(dataloader)
    avg_em      = float(np.mean(all_em))  if all_em else 0.0
    avg_f1      = float(np.mean(all_f1))  if all_f1 else 0.0
    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}

    print(f"\n{'─'*60}")
    print(f"Validation Summary [{VARIANT}]:")
    for k in ['gate_impact', 'gate_variance', 'alpha_diversity', 'memory_contribution']:
        print(f"  {k:25s}: {avg_metrics.get(k, 0.0):.4f}")
    print(f"  {'EM':25s}: {avg_em:.4f}")
    print(f"  {'F1':25s}: {avg_f1:.4f}")
    print(f"{'─'*60}")

    return avg_loss, avg_em, avg_f1, avg_metrics


# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_results(history):
    fig = plt.figure(figsize=(20, 14))
    gs  = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f'AMRPA Ablation — {VARIANT_DESCRIPTIONS[VARIANT]}',
        fontsize=15, fontweight='bold', y=0.98
    )

    COLORS = {
        "full":           "royalblue",
        "no_gate":        "tomato",
        "uniform_memory": "mediumseagreen",
        "no_decay":       "darkorange",
        "fixed_depth":    "mediumpurple",
    }
    c      = COLORS[VARIANT]
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'],   'r-', label='Val',   linewidth=2)
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # EM
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['val_em'], color=c, linewidth=2.5, marker='o')
    ax2.fill_between(epochs, 0, history['val_em'], alpha=0.15, color=c)
    if history['val_em']:
        best = max(history['val_em'])
        ax2.axhline(y=best, color=c, linestyle='--', alpha=0.5, label=f'Best: {best:.3f}')
        ax2.legend()
    ax2.set_title('Exact Match (EM)'); ax2.set_xlabel('Epoch'); ax2.grid(True, alpha=0.3)

    # F1
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history['val_f1'], color=c, linewidth=2.5, marker='s')
    ax3.fill_between(epochs, 0, history['val_f1'], alpha=0.15, color=c)
    if history['val_f1']:
        best = max(history['val_f1'])
        ax3.axhline(y=best, color=c, linestyle='--', alpha=0.5, label=f'Best: {best:.3f}')
        ax3.legend()
    ax3.set_title('F1 Score'); ax3.set_xlabel('Epoch'); ax3.grid(True, alpha=0.3)

    # Claim 1 — Gate Impact
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, history['gate_impact'], color='orange', linewidth=2.5, marker='o')
    ax4.fill_between(epochs, 0, history['gate_impact'], alpha=0.2, color='orange')
    status = "DISABLED (G=1)" if VARIANT == "no_gate" else "ACTIVE"
    ax4.set_title(f'Claim 1 — Gate Impact\n({status})')
    ax4.set_xlabel('Epoch'); ax4.grid(True, alpha=0.3)

    # Claim 2 — Alpha Diversity
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, history['alpha_diversity'], color='teal', linewidth=2.5, marker='s')
    ax5.fill_between(epochs, 0, history['alpha_diversity'], alpha=0.2, color='teal')
    status = "UNIFORM (1/w)" if VARIANT == "uniform_memory" else "MLP-learned"
    ax5.set_title(f'Claim 2 — Alpha Diversity\n({status})')
    ax5.set_xlabel('Epoch'); ax5.grid(True, alpha=0.3)

    # Claim 4 — Memory Contribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(epochs, history['memory_contribution'], color='crimson', linewidth=2.5, marker='^')
    ax6.fill_between(epochs, 0, history['memory_contribution'], alpha=0.2, color='crimson')
    status = "FIXED window=1" if VARIANT == "fixed_depth" else "ADAPTIVE (log2)"
    ax6.set_title(f'Claim 4 — Memory Contribution\n({status})')
    ax6.set_xlabel('Epoch'); ax6.grid(True, alpha=0.3)

    # Claim 3 — Decay Curve
    ax7    = fig.add_subplot(gs[2, 0])
    steps  = np.arange(1, 11)
    decay  = EFFECTIVE_GAMMA ** steps
    ax7.plot(steps, decay, color='darkred', linewidth=3, marker='o', markersize=8)
    ax7.fill_between(steps, 0, decay, alpha=0.25, color='red')
    status = f"DISABLED (γ=1.0)" if VARIANT == "no_decay" else f"ACTIVE (γ={config.GAMMA})"
    ax7.set_title(f'Claim 3 — Memory Decay\n({status})')
    ax7.set_xlabel('Steps back (k)'); ax7.set_ylabel('γ^k'); ax7.grid(True, alpha=0.3)

    # Gate vs F1 dual axis
    if history.get('gate_impact') and history.get('val_f1'):
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.set_xlabel('Epoch'); ax8.set_ylabel('Gate Impact', color='tab:blue')
        l1  = ax8.plot(epochs, history['gate_impact'], color='tab:blue',
                       linewidth=2.5, marker='o', label='Gate Impact')
        ax8.tick_params(axis='y', labelcolor='tab:blue'); ax8.grid(True, alpha=0.3)
        ax8t = ax8.twinx()
        ax8t.set_ylabel('F1', color='tab:green')
        l2   = ax8t.plot(epochs, history['val_f1'], color='tab:green',
                         linewidth=2.5, marker='s', label='F1')
        ax8t.tick_params(axis='y', labelcolor='tab:green')
        ax8.set_title('Gate Mechanism vs F1')
        lines = l1 + l2
        ax8.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=9)

    # EM vs F1
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(epochs, history['val_em'], 'b-', linewidth=2.5, marker='o', label='EM')
    ax9.plot(epochs, history['val_f1'], 'r-', linewidth=2.5, marker='s', label='F1')
    ax9.set_title('QA Performance'); ax9.set_xlabel('Epoch')
    ax9.legend(); ax9.grid(True, alpha=0.3)

    fname = f'amrpa_ablation_{VARIANT}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {fname}")
    plt.show(); plt.close()


# ==============================================================================
# FINAL SUMMARY — copy-paste-ready table row
# ==============================================================================
def print_final_summary(history):
    print("\n" + "=" * 80)
    print(f"ABLATION FINAL SUMMARY — {VARIANT_DESCRIPTIONS[VARIANT]}")
    print("=" * 80)

    best_f1 = max(history['val_f1']) if history['val_f1'] else 0.0
    best_em = max(history['val_em']) if history['val_em'] else 0.0

    print(f"\n  Best Val F1  : {best_f1:.4f}")
    print(f"  Best Val EM  : {best_em:.4f}")
    print(f"  Epochs run   : {len(history['train_loss'])}")

    print(f"\n  Mechanism Averages:")
    for k in ['gate_impact', 'gate_variance', 'alpha_diversity', 'memory_contribution']:
        vals = history.get(k, [])
        if vals:
            print(f"    {k:25s}: mean={np.mean(vals):.4f}  final={vals[-1]:.4f}")

    print("\n  Claim Status:")
    statuses = {
        "Claim 1 Smart Gate":     ("DISABLED (G=1)"    if VARIANT == "no_gate"        else "ACTIVE"),
        "Claim 2 Dynamic Memory": ("DISABLED (uniform)" if VARIANT == "uniform_memory" else "ACTIVE"),
        "Claim 3 Fading Ink":     (f"DISABLED (γ=1.0)" if VARIANT == "no_decay"       else f"ACTIVE (γ={config.GAMMA})"),
        "Claim 4 Adaptive Depth": ("DISABLED (win=1)"  if VARIANT == "fixed_depth"    else "ACTIVE (log2)"),
    }
    for claim, status in statuses.items():
        marker = "✗ OFF" if "DISABLED" in status else "✓ ON "
        print(f"    [{marker}]  {claim:28s} → {status}")

    print("\n  ► Copy into your comparison table:")
    print(f"    | {VARIANT:20s} | F1: {best_f1:.4f} | EM: {best_em:.4f} |")
    print("=" * 80)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    train_ds = PreprocessedQADataset(f"{config.PROCESSED_DATA_DIR}/train_processed.pt")
    val_ds   = PreprocessedQADataset(f"{config.PROCESSED_DATA_DIR}/val_processed.pt")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    model     = RoBERTa_AMRPA_QA(n_amrpa_layers=config.N_AMRPA_LAYERS).to(device)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParams: total={total_p:,}  trainable={trainable_p:,}")

    # Differential learning rates (same as original)
    amrpa_params, qa_head_params, roberta_params = [], [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(k in name for k in ['mlp_alpha', 'w_mem', 'proj_attention', 'gamma_g', 'bias_g']):
                amrpa_params.append(param)
            elif 'qa_outputs' in name:
                qa_head_params.append(param)
            else:
                roberta_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': roberta_params,   'lr': config.LR * 0.5},
        {'params': amrpa_params,     'lr': config.LR * 3},
        {'params': qa_head_params,   'lr': config.LR * 5},
    ], weight_decay=config.WEIGHT_DECAY)

    total_steps  = len(train_loader) * config.EPOCHS
    warmup_steps = max(1, int(total_steps * config.WARMUP_RATIO))
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    history = {
        'train_loss': [], 'val_loss': [], 'val_em': [], 'val_f1': [],
        'gate_impact': [], 'gate_variance': [],
        'alpha_diversity': [], 'memory_contribution': [],
    }

    best_f1          = 0.0
    best_val_loss    = float('inf')
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        print(f"\n{'='*80}\nEPOCH {epoch+1}/{config.EPOCHS}  [variant={VARIANT}]\n{'='*80}")
        t0 = time.time()

        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        val_loss, val_em, val_f1, val_metrics = evaluate(
            model, val_loader, tokenizer, device
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_em'].append(val_em)
        history['val_f1'].append(val_f1)
        for k in ['gate_impact', 'gate_variance', 'alpha_diversity', 'memory_contribution']:
            history[k].append(val_metrics.get(k, 0.0))

        print(f"\n  Epoch {epoch+1}: train={train_loss:.4f}  val={val_loss:.4f}"
              f"  EM={val_em:.4f}  F1={val_f1:.4f}  ({(time.time()-t0)/60:.1f} min)")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f'best_model_{VARIANT}.pt')
            print(f"  ✅ NEW BEST F1: {best_f1:.4f}  → best_model_{VARIANT}.pt")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⏳ Patience {patience_counter}/{config.PATIENCE}")
            if patience_counter >= config.PATIENCE:
                print("  🛑 EARLY STOPPING")
                break

        torch.cuda.empty_cache(); gc.collect()

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE — {VARIANT}")
    print("=" * 80)

    print_final_summary(history)
    plot_results(history)
    return history, model


# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    try:
        history, model = main()
        print(f"\n✅ Done!  Best F1: {max(history['val_f1']):.4f}")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
