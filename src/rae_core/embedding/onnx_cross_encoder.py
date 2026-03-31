import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class OnnxCrossEncoder:
    def __init__(self, model_path: str, tokenizer_path: str):
        # Explicitly limit threads to save memory and CPU
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(length=512)

        self.input_names = [i.name for i in self.session.get_inputs()]
        print(
            f"[OnnxCrossEncoder] Memory-Optimized Init successful. Inputs: {self.input_names}"
        )

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 4) -> np.ndarray:
        all_scores = []

        # Process in small batches to prevent Exit 137 (OOM)
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            batch_input_ids = []
            batch_attention_mask = []
            batch_token_type_ids = []

            for query, doc in batch:
                encoded = self.tokenizer.encode(query, doc)
                batch_input_ids.append(encoded.ids)
                batch_attention_mask.append(encoded.attention_mask)
                batch_token_type_ids.append(encoded.type_ids)

            inputs = {
                "input_ids": np.array(batch_input_ids, dtype=np.int64),
                "attention_mask": np.array(batch_attention_mask, dtype=np.int64),
            }

            if "token_type_ids" in self.input_names:
                inputs["token_type_ids"] = np.array(
                    batch_token_type_ids, dtype=np.int64
                )

            outputs = self.session.run(None, inputs)

            # Extract logits for each item in batch
            batch_logits = outputs[0]
            for logit in batch_logits:
                # Handle different output shapes [1] or [2]
                val = logit
                while isinstance(val, (np.ndarray, list)):
                    val = val[0]
                all_scores.append(float(val))

        return np.array(all_scores)
