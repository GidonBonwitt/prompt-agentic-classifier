import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import re
from transformers import AutoTokenizer
from typing import Tuple

class StudentAgent:
    def __init__(
        self,
        pipeline,                 # your LLM / classifier pipeline
        tokenizer: AutoTokenizer, # tokenizer for the model
        starting_prompt: str,     # “instruction” prefix
        n_iterations: int = 3     # how many self‑reflection loops
    ):
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.starting_prompt = starting_prompt
        self.n_iterations = n_iterations

        # These will be set with set_batch(...)
        self.train_batch: pd.DataFrame | None = None
        self.val_batch:   pd.DataFrame | None = None

        # history of metrics after each accepted update
        self.metrics_history: list[dict] = []

    def set_batch(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ):
        """Store your training & validation DataFrames inside the agent."""
        self.train_batch = train_df.copy().reset_index(drop=True)
        self.val_batch   = val_df.copy().reset_index(drop=True)

    def _test_prompt(self, wrapped_prompt: str) -> Tuple[bool|None, str]:
        """Run the model on one *already‑wrapped* prompt string."""
        out = self.pipeline(
            wrapped_prompt,
            max_new_tokens=60,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=True
        )[0]  # unwrap list → dict

        full = out.get("generated_text", "")
        parts = re.split(r"Your response:\s*", full, maxsplit=1)
        if len(parts) < 2:
            return None, ""
        block = parts[1].strip()
        lines = block.splitlines()
        label_line = lines[0].strip().strip("'\"").lower()
        label = (True if "harmful" in label_line 
                 else False if "benign" in label_line 
                 else None)
        rationale = " ".join(lines[1:]).strip()
        return label, rationale

    def _test_batch(
        self,
        prefix: str = ""
    ) -> pd.DataFrame:
        """
        Classify self.train_batch under the current starting_prompt.
        Returns a new DataFrame with '{prefix}prediction' and '{prefix}response'.
        """
        assert self.train_batch is not None
        # 1) build wrapped prompts
        prompts = [
            self.starting_prompt
            + "\n\nPrompt: “" + txt.strip() + "”"
            + "\nYour response:"
            for txt in self.train_batch['prompt']
        ]

        # 2) batch call
        outputs = self.pipeline(
            prompts,
            max_new_tokens=60,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=True
        )

        preds, resps = [], []
        for item in outputs:
            # unwrap lists
            if isinstance(item, list):
                item = item[0]

            # now item should be a dict
            full = item.get("generated_text", "")
            parts = re.split(r"Your response:\s*", full, maxsplit=1)
            if len(parts) < 2:
                preds.append(None)
                resps.append("")
                continue

            block = parts[1].strip()
            lines = block.splitlines()
            label_line = lines[0].strip().strip("'\"").lower()

            if "harmful" in label_line:
                label = True
            elif "benign" in label_line:
                label = False
            else:
                label = None

            rationale = " ".join(lines[1:]).strip()
            preds.append(label)
            resps.append(rationale)

        # 3) return a new DataFrame
        out = self.train_batch.copy()
        out[f"{prefix}prediction"] = preds
        out[f"{prefix}response"]   = resps

        self.train_batch = out  # update internal state
        return out


    def _call_model(self, prompt_text: str) -> str:
        """Wrapper to call your pipeline."""
        return self.pipeline(prompt_text).strip()   

    def _batch_predict(
        self,
        df: pd.DataFrame,
        prompt_override: str | None = None,
        prefix: str = ""
    ) -> pd.DataFrame:
        """
        Returns a copy of df with two new columns:
        - f"{prefix}prediction"
        - f"{prefix}response"
        based on either self.starting_prompt or prompt_override.
        """
        base = (prompt_override or self.starting_prompt).strip()
        out = df.copy()
        preds, resps = [], []

        for txt in out['prompt']:
            full = (
                base
                + "\n\nPrompt: “" + txt.strip() + "”"
                + "\nYour response:"
            )
            raw = self._call_model(full)
            tok = raw.split()[0].lower() if raw else ""
            if tok == "harmful":
                p = True
            elif tok == "benign":
                p = False
            else:
                p = None

            preds.append(p)
            resps.append(raw)

        out[f"{prefix}prediction"] = preds
        out[f"{prefix}response"]   = resps
        return out


    def classify_batch(self):
        """Overwrite self.train_batch’s main prediction/response columns."""
        assert self.train_batch is not None, "Call set_batch first!"
        pred_df = self._batch_predict(self.train_batch, prefix="")
        # drop any stale cols and assign
        self.train_batch = pred_df.drop(columns=['prediction','response'], errors='ignore')


    def _compute_metrics(self, df: pd.DataFrame, pred_col: str):
        y_true = df['jailbreak'].astype(bool)
        y_pred = df[pred_col].fillna(False)  # None→False for metric calc
        return {
            'precision': precision_score(y_true, y_pred),
            'recall'   : recall_score(y_true, y_pred),
            'f1'       : f1_score(y_true, y_pred),
        }

    def update_batch(self):
        """
        After accepting a candidate, promote its predictions:
          - moves 'candidate_prediction' → 'prediction'
          - moves 'candidate_response'   → 'response'
          - drops the candidate columns
        """
        df = self.train_batch
        df['prediction'] = df['candidate_prediction']
        df['response']   = df['candidate_response']
        df.drop(columns=['candidate_prediction','candidate_response'], inplace=True)

    def learn(self, teacher_feedback: str):
       # ensure baseline exists
        if 'prediction' not in self.train_batch:
            self.classify_batch()
        base_met = self._compute_metrics(self.train_batch, 'prediction')
        best_f1 = base_met['f1']

        for i in range(1, self.n_iterations+1):
            # propose candidate
            candidate = self._self_reflect(teacher_feedback)

            # get candidate outputs (no mutation)
            cand_df = self._batch_predict(self.train_batch, prompt_override=candidate, prefix="candidate_")

            # compute train metrics from candidate_prediction
            met = self._compute_metrics(cand_df, 'candidate_prediction')
            if met['f1'] <= best_f1:
                continue

            # ACCEPT: update starting_prompt, swap into main batch
            self.starting_prompt = candidate
            best_f1 = met['f1']

            # merge candidate cols into self.train_batch
            for col in ['candidate_prediction','candidate_response']:
                self.train_batch[col] = cand_df[col]
            self.update_batch()   # this will promote candidate_ → main prediction

            # now evaluate on val_batch
            val_df = self._batch_predict(self.val_batch, prefix="")
            val_met = self._compute_metrics(val_df, 'prediction')


            # record
            self.metrics_history.append({
                'iter'            : i,
                'train_precision' : met['precision'],
                'train_recall'    : met['recall'],
                'train_f1'        : met['f1'],
                'val_precision'   : val_met['precision'],
                'val_recall'      : val_met['recall'],
                'val_f1'          : val_met['f1'],
            })
            break  # stop after first improvement