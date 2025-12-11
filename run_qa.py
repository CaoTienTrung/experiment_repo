import numpy as np
import pandas as pd

import os, json, logging, argparse, random, pickle, subprocess, math
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.optim import Adam

from transformers import T5Config, AutoTokenizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import ViWordFormerOCNModel
from itertools import islice

# --- Metrics cho explanation ---
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ================== DATA STRUCTURES =====================

class InputExample(object):
    def __init__(self, guid, doc_token, question_text, options, answer=None, explanation=None):
        self.guid = guid
        self.doc_token = doc_token
        self.question_text = question_text
        self.options = options
        self.answer = answer
        self.explanation = explanation  # NEW

    def getQuestion(self):
        return self.question_text

    def getDoctoken(self):
        return self.doc_token

    def getOptions(self):
        return self.options

    def getAnswer(self):
        return self.answer

    def getExplanation(self):
        return self.explanation


class InputFeatures(object):

    def __init__(
        self,
        doc_ids, query_ids, opt_ids,
        doc_mask, query_mask, opt_mask,
        label_id, doc_len, query_len, opt_len,
        guid,
        explanation_ids=None  # NEW
    ):
        self.doc_ids = doc_ids            # [num_labels, max_doc_len]
        self.query_ids = query_ids        # [num_labels, max_query_len]
        self.opt_ids = opt_ids            # [num_labels, max_option_len]
        self.doc_mask = doc_mask
        self.query_mask = query_mask
        self.opt_mask = opt_mask
        self.label_id = label_id
        self.doc_len = doc_len
        self.query_len = query_len
        self.opt_len = opt_len
        self.guid = guid

        self.explanation_ids = explanation_ids  # [max_expl_len]


# ================== DATA LOADING =====================

def read_race_examples(input_data):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for data in input_data:
        doc_id = data['id']
        doc = data["context"].replace('\\n', '\n')
        doc_token = []
        prev_is_whitespace = True
        for c in doc:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_token.append(c)
                else:
                    doc_token[-1] += c
                prev_is_whitespace = False

        explanation = data.get('explanation', None)

        example = InputExample(
            guid=doc_id,
            doc_token=doc_token,
            question_text=data['question'],
            options=data['options'],
            answer=data['answer'],
            explanation=explanation
        )
        examples.append(example)

    return examples


def convert_examples_to_features(
    examples,
    tokenizer,
    max_doc_len,
    max_query_len,
    max_option_len,
    max_expl_len
):
    # ánh xạ label
    label_list = ["A", "B", "C", "D"]
    label_map = {label: i for i, label in enumerate(label_list)}

    pad_id = tokenizer.pad_token_id
    eos_tok = tokenizer.eos_token

    assert max_doc_len   >= 1, "max_doc_len must be >= 1 for EOS"
    assert max_query_len >= 1, "max_query_len must be >= 1 for EOS"
    assert max_option_len>= 1, "max_option_len must be >= 1 for EOS"
    assert max_expl_len  >= 1, "max_expl_len must be >= 1 for EOS"

    features = []

    for example_id, example in enumerate(examples):
        doc_token_full = tokenizer.tokenize(' '.join(example.doc_token))
        query_token_full = tokenizer.tokenize(example.question_text)

        doc_len_list, query_len_list, opt_len_list = [], [], []
        all_doc_ids, all_doc_mask = [], []
        all_query_ids, all_query_mask = [], []
        all_opt_ids, all_opt_mask = [], []

        # ------- DOC / QUERY / OPTION theo từng option -------
        for option_text in example.options:
            opt_token_full = tokenizer.tokenize(option_text)

            # -- DOC --
            real_doc_len  = min(len(doc_token_full), max(0, max_doc_len-1))
            doc_token = doc_token_full[:real_doc_len] + [eos_tok]
            doc_ids = tokenizer.convert_tokens_to_ids(doc_token)
            doc_mask = [1] * len(doc_ids)
            if len(doc_ids) < max_doc_len:
                pad_n = max_doc_len - len(doc_ids)
                doc_ids += [pad_id] * pad_n
                doc_mask += [0] * pad_n

            # -- QUERY --
            real_query_len = min(len(query_token_full), max(0, max_query_len - 1))
            query_token = query_token_full[:real_query_len] + [eos_tok]
            query_ids = tokenizer.convert_tokens_to_ids(query_token)
            query_mask = [1] * len(query_ids)
            if len(query_ids) < max_query_len:
                pad_n = max_query_len - len(query_ids)
                query_ids += [pad_id] * pad_n
                query_mask += [0] * pad_n

            # -- OPTION --
            real_opt_len  = min(len(opt_token_full), max(0, max_option_len - 1))
            opt_tok = opt_token_full[:real_opt_len] + [eos_tok]
            opt_ids = tokenizer.convert_tokens_to_ids(opt_tok)
            opt_mask = [1] * len(opt_ids)
            if len(opt_ids) < max_option_len:
                pad_n = max_option_len - len(opt_ids)
                opt_ids += [pad_id] * pad_n
                opt_mask += [0] * pad_n

            all_doc_ids.append(doc_ids)
            all_query_ids.append(query_ids)
            all_opt_ids.append(opt_ids)

            all_doc_mask.append(doc_mask)
            all_query_mask.append(query_mask)
            all_opt_mask.append(opt_mask)

            doc_len_list.append(real_doc_len)
            query_len_list.append(real_query_len)
            opt_len_list.append(real_opt_len)

        label_id = label_map[example.answer]

        # ------- EXPLANATION (một câu cho cả sample) -------
        expl_tokens = tokenizer.tokenize(example.explanation)
        real_expl_len = min(len(expl_tokens), max(0, max_expl_len - 1))
        expl_tokens = expl_tokens[:real_expl_len] + [eos_tok]
        expl_ids = tokenizer.convert_tokens_to_ids(expl_tokens)

        if len(expl_ids) < max_expl_len:
            pad_n = max_expl_len - len(expl_ids)
            expl_ids += [pad_id] * pad_n
        else:
            expl_ids = expl_ids[:max_expl_len]

        features.append(
            InputFeatures(
                doc_ids=all_doc_ids,
                query_ids=all_query_ids,
                opt_ids=all_opt_ids,
                doc_mask=all_doc_mask,
                query_mask=all_query_mask,
                opt_mask=all_opt_mask,
                label_id=label_id,
                doc_len=doc_len_list,
                query_len=query_len_list,
                opt_len=opt_len_list,
                guid=example.guid,
                explanation_ids=expl_ids,
            )
        )

    return features


def load_data(
    data_dir,
    tokenizer,
    max_doc_len=400,
    max_query_len=80,
    max_option_len=20,
    max_expl_len=64,
    is_training=True,
    max_samples=None,  
):
    if is_training:
        subset_list = ['train', 'dev']
    else:
        subset_list = ['test']

    examples, features = {}, {}
    for subset in subset_list:
        file_path = os.path.join(data_dir, f"{subset}_MCQA_VietBGE.json")

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        items = list(data.items())
        if max_samples is not None:
            items = items[:max_samples]

        alldata = []
        for _id, id_items in items:
            context = id_items['context']
            question = id_items['question']
            options = id_items['answer_options']
            answer = id_items['correct_answer']
            explanation = id_items.get('explanation', None)

            alldata.append({
                'id': _id,
                'context': context,
                'question': question,
                'options': options,
                'answer': answer,
                'explanation': explanation,
            })

        examples[subset] = read_race_examples(alldata)
        features[subset] = convert_examples_to_features(
            examples[subset],
            tokenizer,
            max_doc_len,
            max_query_len,
            max_option_len,
            max_expl_len,
        )

    return examples, features


# ================== TRAIN / EVAL =====================

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")


def softmax_np(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


# ------- Explanation metrics -------

def evaluate_explanation_metrics(outputs_json):
    """
    outputs_json[guid] = {
        "pred_explanation": str,
        "gold_explanation": str or None
    }
    Trả về: BLEU-4, ROUGE-L (F1, P, R).
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method1

    rougeL_f_list, rougeL_p_list, rougeL_r_list = [], [], []
    bleu4_list = []

    for guid, item in outputs_json.items():
        ref = item.get("gold_explanation", None)
        pred = item.get("pred_explanation", "")

        if not ref:
            continue

        ref = ref.strip()
        pred = pred.strip()
        if len(ref) == 0 or len(pred) == 0:
            continue

        # ROUGE-L
        r = scorer.score(ref, pred)['rougeL']
        rougeL_f_list.append(r.fmeasure)
        rougeL_p_list.append(r.precision)
        rougeL_r_list.append(r.recall)

        # BLEU-4
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        bleu4 = sentence_bleu(
            [ref_tokens],
            pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie
        )
        bleu4_list.append(bleu4)

    def avg(x):
        return float(np.mean(x)) if len(x) > 0 else 0.0

    metrics = {
        "bleu4": avg(bleu4_list),
        "rougeL_f": avg(rougeL_f_list),
        "rougeL_p": avg(rougeL_p_list),
        "rougeL_r": avg(rougeL_r_list),
    }

    print("Explanation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


# def evaluate_explanation_metrics(outputs_json):
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#     smoothie = SmoothingFunction().method1

#     rouge1_f, rougeL_f = [], []
#     bleu1, bleu2, bleu3, bleu4 = [], [], [], []

#     for guid, item in outputs_json.items():
#         ref = item.get("ref_explanation", None)
#         pred = item.get("pred_explanation", "")

#         if not ref:
#             continue

#         ref = ref.strip()
#         pred = pred.strip()
#         if len(ref) == 0 or len(pred) == 0:
#             continue

#         # ROUGE
#         r = scorer.score(ref, pred)
#         rouge1_f.append(r['rouge1'].fmeasure)
#         rougeL_f.append(r['rougeL'].fmeasure)

#         # BLEU
#         ref_tokens = ref.split()
#         pred_tokens = pred.split()

#         bleu1.append(sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie))
#         bleu2.append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
#         bleu3.append(sentence_bleu([ref_tokens], pred_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothie))
#         bleu4.append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

#     def avg(x):
#         return float(np.mean(x)) if len(x) > 0 else 0.0

#     metrics = {
#         "rouge1_f": avg(rouge1_f),
#         "rougeL_f": avg(rougeL_f),
#         "bleu1": avg(bleu1),
#         "bleu2": avg(bleu2),
#         "bleu3": avg(bleu3),
#         "bleu4": avg(bleu4),
#     }

#     print("Explanation metrics:")
#     for k, v in metrics.items():
#         print(f"  {k}: {v:.4f}")

#     return metrics


def dev_evaluate(
    model,
    dataloader,
    device,
    dev_examples,
    tokenizer,
    max_expl_len,
):
    """
    Dev eval: 1 pass cho cả classification + explanation.
    dataloader phải yield: (..., expl_labels, example_indexes)
    """
    model.eval()

    y_pred, y_true = [], []

    # map index -> example để lấy gold explanation string
    idx2example = {i: ex for i, ex in enumerate(dev_examples)}

    outputs_json = {}
    label_list = ["A", "B", "C", "D"]

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        (
            doc_ids, doc_mask,
            query_ids, query_mask,
            opt_ids, opt_mask,
            label_ids,
            expl_labels,          # không dùng trực tiếp ở eval
            example_indexes,
        ) = batch

        with torch.no_grad():
            outputs = model(
                doc_ids=doc_ids,
                doc_mask=doc_mask,
                query_ids=query_ids,
                query_mask=query_mask,
                opt_ids=opt_ids,
                opt_mask=opt_mask,
                labels=label_ids,
                explanation_labels=expl_labels,
                return_dict=True,
            )
            logits = outputs.logits

            # classification
            logits_np = logits.detach().cpu().numpy()
            logits_np = softmax_np(logits_np)
            predicted_labels = np.argmax(logits_np, axis=1)
            y_pred.extend(predicted_labels)
            label_ids_np = label_ids.to('cpu').numpy()
            y_true.extend(label_ids_np)

            # explanation generation dùng predicted labels
            pred_idx_tensor = torch.tensor(predicted_labels, dtype=torch.long, device=device)
            explanations, logits = model.generate_explanation(
                doc_ids=doc_ids,
                doc_mask=doc_mask,
                query_ids=query_ids,
                query_mask=query_mask,
                opt_ids=opt_ids,
                opt_mask=opt_mask,
                chosen_idx=pred_idx_tensor,
                max_length=max_expl_len,
                tokenizer=tokenizer,
                device=device,
            )

        for i, example_idx in enumerate(example_indexes):
            idx = example_idx.item()
            example = idx2example[idx]
            guid = example.guid
            gold_expl = example.getExplanation()

            outputs_json[guid] = {
                "pred_answer": label_list[predicted_labels[i]],
                "gold_answer": label_list[label_ids_np[i]],
                "pred_explanation": explanations[i],
                "gold_explanation": gold_expl,
            }

    accuracy = accuracy_score(np.array(y_true), np.array(y_pred))
    f1_macro = f1_score(np.array(y_true), np.array(y_pred), average='macro')

    expl_metrics = evaluate_explanation_metrics(outputs_json)

    return accuracy, f1_macro, expl_metrics


def train(
    train_loader,
    dev_loader,
    dev_examples,
    model,
    optimizer,
    device,
    epochs,
    checkpoint_dir,
    tokenizer,
    max_expl_len,
    eval_start_epoch=1,
    eval_interval=1,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    log_file = os.path.join(checkpoint_dir, "train.log")

    # Nếu chưa có thì tạo file + header
    if not os.path.exists(log_file):
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(
                "epoch\trun_loss\tdev_acc\tdev_f1_macro\t"
                "bleu4\trougeL_f\trougeL_p\trougeL_r\n"
            )

    best_dev_acc = -1.0
    best_bleu4 = -1.0
    best_rougelf1 = -1.0
    best_acc_ckpt_path = os.path.join(checkpoint_dir, "best_acc.pth")
    best_bleu_ckpt_path = os.path.join(checkpoint_dir, "best_bleu.pth")
    best_rougelf1_ckpt_path = os.path.join(checkpoint_dir, "best_rougelf1.pth")

    for epoch in range(epochs):
        model.train()
        run_loss = 0.0
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for step, batch in progress_bar:
            batch = tuple(t.to(device) for t in batch)
            (
                doc_ids, doc_mask,
                query_ids, query_mask,
                opt_ids, opt_mask,
                label_ids,
                expl_labels,
            ) = batch

            outputs = model(
                doc_ids=doc_ids,
                doc_mask=doc_mask,
                query_ids=query_ids,
                query_mask=query_mask,
                opt_ids=opt_ids,
                opt_mask=opt_mask,
                labels=label_ids,
                explanation_labels=expl_labels,
                return_dict=True,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            avg_batch_loss = run_loss / (step + 1)
            progress_bar.set_postfix({"loss": f"{avg_batch_loss:.4f}"})

        current_epoch = epoch + 1
        print(f"[Epoch {current_epoch}] train_loss: {run_loss:.4f}")
        
        if (
            current_epoch >= eval_start_epoch
            and (current_epoch - eval_start_epoch) % eval_interval == 0
        ):
            dev_acc, dev_f1, expl_metrics = dev_evaluate(
                model,
                dev_loader,
                device,
                dev_examples,
                tokenizer,
                max_expl_len,
            )
            print(f"[Epoch {epoch + 1}] train_loss: {run_loss:.4f}")
            print(f"  Dev acc: {dev_acc:.4f}, Dev f1_macro: {dev_f1:.4f}")
            print("  Dev explanation metrics:")
            for k, v in expl_metrics.items():
                print(f"    {k}: {v:.4f}")
                
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{current_epoch}\t"
                    f"{run_loss:.6f}\t"
                    f"{dev_acc:.6f}\t"
                    f"{dev_f1:.6f}\t"
                    f"{expl_metrics.get('bleu4', 0.0):.6f}\t"
                    f"{expl_metrics.get('rougeL_f', 0.0):.6f}\t"
                    f"{expl_metrics.get('rougeL_p', 0.0):.6f}\t"
                    f"{expl_metrics.get('rougeL_r', 0.0):.6f}\n"
                )

            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
            save_checkpoint(model, optimizer, epoch + 1, run_loss, checkpoint_path)
            
                # ==== Best checkpoint theo ACC ====
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_checkpoint(model, optimizer, current_epoch, run_loss, best_acc_ckpt_path)
                print(
                    f"BEST ACC: {best_dev_acc:.4f} "
                    f"at epoch {current_epoch}, saved to {best_acc_ckpt_path}"
                )

            # ==== Best checkpoint theo BLEU-4 ====
            bleu4 = expl_metrics.get("bleu4", 0.0)
            if bleu4 > best_bleu4:
                best_bleu4 = bleu4
                save_checkpoint(model, optimizer, current_epoch, run_loss, best_bleu_ckpt_path)
                print(
                    f"BEST BLEU4: {best_bleu4:.4f} "
                    f"at epoch {current_epoch}, saved to {best_bleu_ckpt_path}"
                )
                
            # ==== Best checkpoint theo RougeL-f1 ====
            rougelf1 = expl_metrics.get('rougeL_f', 0.0)
            if rougelf1 > best_rougelf1:
                best_rougelf1 = rougelf1
                save_checkpoint(model, optimizer, current_epoch, run_loss, best_rougelf1_ckpt_path)
                print(
                    f"BEST BLEU4: {best_rougelf1:.4f} "
                    f"at epoch {current_epoch}, saved to {best_rougelf1_ckpt_path}"
                )


def test_evaluate(model, features, examples, batch_size, device, tokenizer, max_expl_len):
    """
    Đánh giá trên test:
      - accuracy, macro-F1 answer
      - full outputs_json (answer + explanation)
      - metrics explanation
    """
    all_doc_ids = torch.tensor([f.doc_ids for f in features], dtype=torch.long)
    all_doc_mask = torch.tensor([f.doc_mask for f in features], dtype=torch.long)
    all_query_ids = torch.tensor([f.query_ids for f in features], dtype=torch.long)
    all_query_mask = torch.tensor([f.query_mask for f in features], dtype=torch.long)
    all_opt_ids = torch.tensor([f.opt_ids for f in features], dtype=torch.long)
    all_opt_mask = torch.tensor([f.opt_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_doc_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(
        all_doc_ids, all_doc_mask,
        all_query_ids, all_query_mask,
        all_opt_ids, all_opt_mask,
        all_label_ids, all_example_index
    )

    eval_dataloader = DataLoader(eval_data, batch_size=batch_size)

    model.eval()
    y_pred, y_true = [], []

    idx2example = {i: ex for i, ex in enumerate(examples)}
    label_list = ["A", "B", "C", "D"]
    outputs_json = {}

    for batch in tqdm(eval_dataloader, desc="Test eval", unit="batch"):
        batch = tuple(t.to(device) for t in batch)
        doc_ids, doc_mask, query_ids, query_mask, opt_ids, opt_mask, label_ids, example_indexes = batch

        with torch.no_grad():
            outputs = model(
                doc_ids=doc_ids,
                doc_mask=doc_mask,
                query_ids=query_ids,
                query_mask=query_mask,
                opt_ids=opt_ids,
                opt_mask=opt_mask,
                labels=label_ids,
                return_dict=True,
            )
            logits = outputs.logits

            logits_np = logits.detach().cpu().numpy()
            logits_np = softmax_np(logits_np)
            predicted_labels = np.argmax(logits_np, axis=1)
            y_pred.extend(predicted_labels)
            label_ids_np = label_ids.to('cpu').numpy()
            y_true.extend(label_ids_np)

            pred_idx_tensor = torch.tensor(predicted_labels, dtype=torch.long, device=device)
            explanations, logits = model.generate_explanation(
                doc_ids=doc_ids,
                doc_mask=doc_mask,
                query_ids=query_ids,
                query_mask=query_mask,
                opt_ids=opt_ids,
                opt_mask=opt_mask,
                chosen_idx=pred_idx_tensor,
                max_length=max_expl_len,
                tokenizer=tokenizer,
                device=device,
            )

        for i, example_idx in enumerate(example_indexes):
            idx = example_idx.item()
            ex = idx2example[idx]
            guid = ex.guid
            gold_expl = ex.getExplanation()

            outputs_json[guid] = {
                "pred_answer": label_list[predicted_labels[i]],
                "gold_answer": label_list[label_ids_np[i]],
                "pred_explanation": explanations[i],
                "gold_explanation": gold_expl,
            }

    accuracy = accuracy_score(np.array(y_true), np.array(y_pred))
    f1_macro = f1_score(np.array(y_true), np.array(y_pred), average='macro')

    expl_metrics = evaluate_explanation_metrics(outputs_json)

    return outputs_json, accuracy, f1_macro, expl_metrics



def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def save_json(obj, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    return model


# ----------------------------- Args & main ------------------------- #
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="VietAI/vit5-base")
    parser.add_argument("--tokenizer_name", type=str, default="VietAI/vit5-base")

    parser.add_argument("--data_dir", type=str, default=r"F:\Deep_learning\PROJECT\Dataset")
    parser.add_argument("--checkpoint_dir", type=str, default=r"F:\Deep_learning\PROJECT\checkpoint")
    parser.add_argument("--model_pth", type=str, default=r"model_epoch_1.pth")
    parser.add_argument("--output_dir", type=str, default=r"F:\Deep_learning\PROJECT\Dataset\Result")

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_batch_size", type=int, default=12)
    parser.add_argument("--eval_batch_size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max_doc_len", type=int, default=400)
    parser.add_argument("--max_query_len", type=int, default=80)
    parser.add_argument("--max_option_len", type=int, default=20)
    parser.add_argument("--max_expl_len", type=int, default=64)
    parser.add_argument("--num_labels", type=int, default=4)
    
    parser.add_argument(
        "--eval_start_epoch",
        type=int,
        default=1,
        help="Epoch (1-based) to start dev evaluation."
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1,
        help="Run dev evaluation every N epochs starting from eval_start_epoch."
    )

    parser.add_argument(
        "--do_train",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Flag to do training"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Giới hạn số sample mỗi split (debug). None = dùng full."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    config = T5Config.from_pretrained(args.model_name_or_path)

    print("Complete loading pretrained config & tokenizer ...\n")

    model = ViWordFormerOCNModel.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        config=config,
        num_labels=args.num_labels,
        max_doc_len=args.max_doc_len,
        max_query_len=args.max_query_len,
        max_option_len=args.max_option_len,
        explanation_loss_weight=1.0,  # có thể chỉnh
    )
    model.to(device)
    print("Complete initializing ViWordFormerOCNModel ...\n")

    data_dir = args.data_dir

    if args.do_train == "True":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        examples, features = load_data(
            data_dir,
            tokenizer,
            max_doc_len=args.max_doc_len,
            max_query_len=args.max_query_len,
            max_option_len=args.max_option_len,
            max_expl_len=args.max_expl_len,
            is_training=True,
            max_samples=args.max_samples,
        )

        print("Complete creating features ...\n")

        pad_id = tokenizer.pad_token_id

        # ----- TRAIN SET -----
        train_features = features['train']
        all_doc_ids = torch.tensor([f.doc_ids for f in train_features], dtype=torch.long)
        all_doc_mask = torch.tensor([f.doc_mask for f in train_features], dtype=torch.long)
        all_query_ids = torch.tensor([f.query_ids for f in train_features], dtype=torch.long)
        all_query_mask = torch.tensor([f.query_mask for f in train_features], dtype=torch.long)
        all_opt_ids = torch.tensor([f.opt_ids for f in train_features], dtype=torch.long)
        all_opt_mask = torch.tensor([f.opt_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        all_expl_ids = torch.tensor([f.explanation_ids for f in train_features], dtype=torch.long)
        all_expl_labels = all_expl_ids.clone()
        all_expl_labels[all_expl_ids == pad_id] = -100

        train_example_index = torch.arange(all_doc_ids.size(0), dtype=torch.long)

        train_data = TensorDataset(
            all_doc_ids, all_doc_mask,
            all_query_ids, all_query_mask,
            all_opt_ids, all_opt_mask,
            all_label_ids,
            all_expl_labels,
        )
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)

        # ----- DEV SET -----
        dev_features = features['dev']
        dev_doc_ids = torch.tensor([f.doc_ids for f in dev_features], dtype=torch.long)
        dev_doc_mask = torch.tensor([f.doc_mask for f in dev_features], dtype=torch.long)
        dev_query_ids = torch.tensor([f.query_ids for f in dev_features], dtype=torch.long)
        dev_query_mask = torch.tensor([f.query_mask for f in dev_features], dtype=torch.long)
        dev_opt_ids = torch.tensor([f.opt_ids for f in dev_features], dtype=torch.long)
        dev_opt_mask = torch.tensor([f.opt_mask for f in dev_features], dtype=torch.long)
        dev_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)

        dev_expl_ids = torch.tensor([f.explanation_ids for f in dev_features], dtype=torch.long)
        dev_expl_labels = dev_expl_ids.clone()
        dev_expl_labels[dev_expl_ids == pad_id] = -100

        dev_example_index = torch.arange(dev_doc_ids.size(0), dtype=torch.long)

        dev_data = TensorDataset(
            dev_doc_ids, dev_doc_mask,
            dev_query_ids, dev_query_mask,
            dev_opt_ids, dev_opt_mask,
            dev_label_ids,
            dev_expl_labels,
            dev_example_index,
        )
        dev_dataloader = DataLoader(dev_data, batch_size=args.eval_batch_size)

        dev_examples = examples['dev']

        train(
            train_dataloader,
            dev_dataloader,
            dev_examples,
            model,
            optimizer,
            device,
            args.num_train_epochs,
            args.checkpoint_dir,
            tokenizer,
            args.max_expl_len,
            eval_start_epoch=args.eval_start_epoch,
            eval_interval=args.eval_interval,
        )

    if args.do_train == "False":
        model = load_checkpoint(model, os.path.join(args.checkpoint_dir, args.model_pth), device)
        examples, features = load_data(
            data_dir,
            tokenizer,
            max_doc_len=args.max_doc_len,
            max_query_len=args.max_query_len,
            max_option_len=args.max_option_len,
            max_expl_len=args.max_expl_len,
            is_training=False,
            max_samples=args.max_samples,
        )

        test_features = features['test']
        test_examples = examples['test']

        os.makedirs(args.output_dir, exist_ok=True)

        # Test eval: answer + explanation
        outputs_json, accuracy, f1_macro, expl_metrics = test_evaluate(
            model,
            test_features,
            test_examples,
            args.eval_batch_size,
            device,
            tokenizer,
            args.max_expl_len,
        )

        print(f"[TEST] Accuracy: {accuracy:.4f}")
        print(f"[TEST] f1_macro: {f1_macro:.4f}")

        print("[TEST] Explanation metrics:")
        for k, v in expl_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Lưu FULL: answer + pred_explanation + gold_explanation
        save_json(outputs_json, os.path.join(args.output_dir, "pred_test_with_expl.json"))

        # Lưu metrics explanation
        metrics_path = os.path.join(args.output_dir, "test_explanation_metrics.json")
        save_json(expl_metrics, metrics_path)
        print(f"Saved test explanation metrics to {metrics_path}")


if __name__ == "__main__":
    main()
