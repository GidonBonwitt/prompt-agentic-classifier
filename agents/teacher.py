import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score, f1_score
import re
from transformers import AutoTokenizer
from typing import Tuple
from typing import Dict, Any

class TeacherAgent:
    def __init__(
        self,
        model_pipeline,         # your HF pipeline or similar
        tokenizer,
        starting_prompt: str,    # template with {CURRENT_AGENT_PROMPT} and {TEST_RESULTS}
        student_prompt: str
    ):
        self.pipe = model_pipeline
        self.tokenizer = tokenizer
        self.starting_prompt = starting_prompt
        self.student_prompt = student_prompt

    def review(self, student_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Reviews all misclassifications, not just two, and extracts
        systemic weaknesses.
        """
        # 1) Compute basic error statistics
        total     = len(student_df)
        errors_df = student_df[ student_df['prediction'] != student_df['jailbreak'] ]
        n_errors  = len(errors_df)
        fp        = len(errors_df[(errors_df['prediction'] == True)  & (errors_df['jailbreak'] == False)])
        fn        = len(errors_df[(errors_df['prediction'] == False) & (errors_df['jailbreak'] == True)])

        # 2) Sample up to 10 misclassified prompts for context
        sample_df = errors_df.sample(n=min(10, n_errors), random_state=42)
        sample_md = sample_df.rename(columns={
                        'jailbreak':    'true_label',
                        'prediction':   'predicted_label',
                        'response':     'reason'
                     })[['prompt','true_label','predicted_label','reason']] \
                     .to_markdown(index=False)

        # 3) Build a richer TEST_RESULTS block
        test_summary = (
            f"- Total examples: {total}\n"
            f"- Misclassifications: {n_errors} ({n_errors/total:.1%} error rate)\n"
            f"  - False Positives: {fp}\n"
            f"  - False Negatives: {fn}\n\n"
            "**Sample Misclassified Prompts:**\n" + sample_md
        )

        # 4) Fill in your prompt template
        prompt = (
            self.starting_prompt
            .replace("{CURRENT_AGENT_PROMPT}", self.student_prompt)
            .replace("{TEST_RESULTS}", test_summary)
        )

        # 5) Call the LLM
        raw = self.pipe(
            prompt,
            max_new_tokens=300,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=True
        )
        if isinstance(raw, list):
            raw = raw[0].get("generated_text", "")

       # Split off everything before the final '---' line
        parts = re.split(r'\n-{3,}\n', raw)
        body = parts[-1]  # the LLM’s actual response

        # 6) Parse the sections from `body`
        def extract(text: str, section: str) -> str:
            """
            Finds **section** in text and returns the text up to the next section or end.
            """
            pattern = rf"\*\*{re.escape(section)}\*\*\s*(.*?)(?=\n\*\*|$)"
            m = re.search(pattern, text, flags=re.DOTALL|re.IGNORECASE)
            return m.group(1).strip() if m else ""

        return {
            "raw_output" : raw,
            "weakness"   : extract(body, "Weakness Identified"),
            "example"    : extract(body, "Example"),
            "suggestion" : extract(body, "Suggested Prompt Improvement")
        }