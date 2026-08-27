import os
import sys
import time
import asyncio
import threading
import numpy as np
from typing import List, Dict, Optional, Any, AsyncGenerator

class BatchRequest:
    def __init__(self, request_id: str, prompt: str, token_ids: List[int], max_tokens: int = 512, temperature: float = 0.0):
        self.request_id = request_id
        self.prompt = prompt
        self.token_ids = token_ids
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stream_queue: asyncio.Queue = asyncio.Queue()
        self.output_tokens: List[int] = []
        self.done = False
        self.finish_reason = None
        self.created_at = time.time()

class DynamicBatchEngine:
    """
    High-Performance Dynamic Micro-Batching Engine for CPU / 2-vCPU Cloud Environments.
    
    Coalesces concurrent generation requests within a micro-window (e.g. 2-5ms) into
    a single vectorized batch (GEMM) utilizing AVX2/AVX-512 SIMD execution units.
    """
    _instance = None

    def __init__(
        self,
        model=None,
        tokenizer=None,
        max_batch_size: int = 16,
        batch_timeout_ms: float = 3.0,
        eos_tokens: Optional[set] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = int(os.environ.get("SLM_MAX_BATCH_SIZE", max_batch_size))
        self.batch_timeout_ms = float(os.environ.get("SLM_BATCH_TIMEOUT_MS", batch_timeout_ms))
        self.request_queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.total_processed_requests = 0
        self.total_batches = 0
        
        self.eos_tokens = eos_tokens or {
            151643, 151645, 248046, 248044, 248045, # Qwen 2.5/3.5 ChatML (<|im_end|>, <|endoftext|>)
            32000, 32007, 107, 128001, 128009       # Llama / Phi / Mistral EOS tokens
        }

    @classmethod
    def get_instance(cls, model=None, tokenizer=None):
        if cls._instance is None:
            cls._instance = cls(model=model, tokenizer=tokenizer)
        elif model is not None and cls._instance.model is None:
            cls._instance.model = model
        if tokenizer is not None and cls._instance.tokenizer is None:
            cls._instance.tokenizer = tokenizer
        return cls._instance

    def ensure_worker(self):
        if self.worker_task is None or self.worker_task.done():
            try:
                loop = asyncio.get_running_loop()
                self.worker_task = loop.create_task(self._batch_worker_loop())
                self.is_running = True
            except RuntimeError:
                pass

    async def generate_stream(
        self,
        prompt: str,
        token_ids: Optional[List[int]] = None,
        max_tokens: int = 512,
        request_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronous generator yielding streamed tokens for a single request,
        while executing in a vectorized dynamic batch under the hood.
        """
        self.ensure_worker()
        
        if token_ids is None and self.tokenizer is not None:
            token_ids = self.tokenizer.encode(prompt)
            if hasattr(token_ids, "ids"):
                token_ids = token_ids.ids

        if not token_ids:
            return

        req_id = request_id or f"req_{time.time()}_{id(token_ids)}"
        req = BatchRequest(
            request_id=req_id,
            prompt=prompt,
            token_ids=list(token_ids),
            max_tokens=max_tokens
        )

        await self.request_queue.put(req)

        while True:
            token_str = await req.stream_queue.get()
            if token_str is None:
                break
            yield token_str

    async def _batch_worker_loop(self):
        """
        Continuous worker loop collecting requests within batch_timeout_ms window.
        """
        while True:
            try:
                first_req = await self.request_queue.get()
                batch: List[BatchRequest] = [first_req]

                # Micro-batching window: Collect additional concurrent requests if available
                deadline = asyncio.get_event_loop().time() + (self.batch_timeout_ms / 1000.0)
                while len(batch) < self.max_batch_size:
                    remaining_time = deadline - asyncio.get_event_loop().time()
                    if remaining_time <= 0:
                        break
                    try:
                        next_req = await asyncio.wait_for(self.request_queue.get(), timeout=remaining_time)
                        batch.append(next_req)
                    except asyncio.TimeoutError:
                        break

                if batch:
                    self.total_batches += 1
                    self.total_processed_requests += len(batch)
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[DynamicBatchEngine] Worker error: {e}")
                await asyncio.sleep(0.01)

    async def _process_batch(self, batch: List[BatchRequest]):
        """
        Executes parallel multi-sequence prefill and batched iterative decoding.
        """
        if not batch or self.model is None:
            for req in batch:
                await req.stream_queue.put(None)
            return

        loop = asyncio.get_running_loop()
        
        # Run batched ONNX inference in threadpool executor to keep asyncio loop responsive
        await loop.run_in_executor(None, self._execute_batch_sync, batch, loop)

    def _execute_batch_sync(self, batch: List[BatchRequest], loop: asyncio.AbstractEventLoop):
        B = len(batch)
        if B == 0:
            return

        # Check if single request fast-path or multi-request batching
        lengths = [len(r.token_ids) for r in batch]
        max_prompt_len = max(lengths)
        max_gen_tokens = min(max(r.max_tokens for r in batch), 2048)

        # Right-pad prompts to max_prompt_len
        pad_id = 0
        input_ids = np.full((B, max_prompt_len), pad_id, dtype=np.int64)
        attention_mask = np.zeros((B, max_prompt_len), dtype=np.int64)

        for i, req in enumerate(batch):
            input_ids[i, :lengths[i]] = req.token_ids
            attention_mask[i, :lengths[i]] = 1

        active_mask = [True] * B
        cur_lengths = list(lengths)

        # Build initial dec_inputs
        model = self.model
        dec_inputs = {}

        if "inputs_embeds" in model.dec_input_names:
            embed_out = model.embed_sess.run(None, {"input_ids": input_ids})[0]
            dec_inputs["inputs_embeds"] = embed_out
            dec_inputs["attention_mask"] = attention_mask
        else:
            dec_inputs["input_ids"] = input_ids
            dec_inputs["attention_mask"] = attention_mask

        # Position IDs
        if "position_ids" in model.dec_input_names:
            pos_ids_list = []
            for i in range(B):
                pos_ids_list.append(np.arange(max_prompt_len, dtype=np.int64))
            pos_ids_arr = np.array(pos_ids_list, dtype=np.int64) # (B, max_prompt_len)
            pos_ids = np.repeat(np.expand_dims(pos_ids_arr, 0), 3, axis=0) # (3, B, max_prompt_len)
            dec_inputs["position_ids"] = pos_ids

        for inp in model.dec_sess.get_inputs():
            if inp.name not in dec_inputs:
                if inp.name == "num_logits_to_keep":
                    dec_inputs[inp.name] = np.array(1, dtype=np.int64)
                else:
                    shape = []
                    for idx, d in enumerate(inp.shape):
                        if idx == 0 and isinstance(d, str):
                            shape.append(B)
                        elif isinstance(d, int):
                            shape.append(d)
                        else:
                            shape.append(0 if "sequence" in str(d) else 1)
                    dtype = np.int64 if "int64" in str(inp.type) else (np.float16 if "float16" in str(inp.type) else np.float32)
                    dec_inputs[inp.name] = np.zeros(shape, dtype=dtype)

        # Prefill forward pass (All B sequences vectorized at once with AVX SIMD)
        try:
            last_outputs = model.dec_sess.run(None, dec_inputs)
        except Exception as e:
            # Fallback per-item execution if model graph requires strict B=1 shapes
            print(f"[DynamicBatchEngine] Batched prefill note ({e}), running individual stream dispatch")
            self._execute_fallback_sync(batch, loop)
            return

        logits = last_outputs[0] # (B, seq_len, vocab_size) or (B, 1, vocab_size)

        # Iterative batched decode loop
        step = 0
        next_tokens = np.zeros((B, 1), dtype=np.int64)

        for i in range(B):
            tok = int(np.argmax(logits[i, -1, :]))
            next_tokens[i, 0] = tok
            req = batch[i]
            req.output_tokens.append(tok)
            tok_text = self._decode_token(tok)
            if self._is_eos(tok, tok_text) or len(req.output_tokens) >= req.max_tokens:
                active_mask[i] = False
                req.done = True
                loop.call_soon_threadsafe(req.stream_queue.put_nowait, None)
            else:
                loop.call_soon_threadsafe(req.stream_queue.put_nowait, tok_text)

        step += 1

        while any(active_mask) and step < max_gen_tokens:
            # Update KV cache inputs from previous step
            for idx, past_name in model.kv_mappings:
                dec_inputs[past_name] = last_outputs[idx]

            # Prepare single-token inputs for active batch
            if "inputs_embeds" in model.dec_input_names:
                next_embed = model.embed_sess.run(None, {"input_ids": next_tokens})[0]
                dec_inputs["inputs_embeds"] = next_embed
            else:
                dec_inputs["input_ids"] = next_tokens

            # Update attention mask and position IDs
            cur_max_len = max(cur_lengths) + step
            att_mask = np.ones((B, cur_max_len), dtype=np.int64)
            dec_inputs["attention_mask"] = att_mask

            if "position_ids" in model.dec_input_names:
                step_pos = np.array([[cur_lengths[i] + step - 1] for i in range(B)], dtype=np.int64)
                dec_inputs["position_ids"] = np.repeat(np.expand_dims(step_pos, 0), 3, axis=0)

            last_outputs = model.dec_sess.run(None, dec_inputs)
            logits = last_outputs[0]

            for i in range(B):
                if not active_mask[i]:
                    continue
                tok = int(np.argmax(logits[i, -1, :]))
                next_tokens[i, 0] = tok
                req = batch[i]
                req.output_tokens.append(tok)
                tok_text = self._decode_token(tok)

                if self._is_eos(tok, tok_text) or len(req.output_tokens) >= req.max_tokens:
                    active_mask[i] = False
                    req.done = True
                    loop.call_soon_threadsafe(req.stream_queue.put_nowait, None)
                else:
                    loop.call_soon_threadsafe(req.stream_queue.put_nowait, tok_text)

            step += 1

        # Ensure any remaining streams receive termination None
        for i, req in enumerate(batch):
            if not req.done:
                req.done = True
                loop.call_soon_threadsafe(req.stream_queue.put_nowait, None)

    def _execute_fallback_sync(self, batch: List[BatchRequest], loop: asyncio.AbstractEventLoop):
        """Fallback to high-speed sequential worker if dynamic shape is rigid."""
        from main import Qwen35ONNXGenerator
        for req in batch:
            gen = Qwen35ONNXGenerator(self.model)
            gen.append_tokens(req.token_ids)
            while not gen.is_done():
                gen.generate_next_token()
                toks = gen.get_next_tokens()
                if toks:
                    tok = toks[0]
                    tok_text = self._decode_token(tok)
                    loop.call_soon_threadsafe(req.stream_queue.put_nowait, tok_text)
                else:
                    break
            loop.call_soon_threadsafe(req.stream_queue.put_nowait, None)

    def _decode_token(self, token_id: int) -> str:
        if self.tokenizer is None:
            return ""
        try:
            return self.tokenizer.decode([token_id])
        except Exception:
            return ""

    def _is_eos(self, token_id: int, token_text: str) -> bool:
        if token_id in self.eos_tokens:
            return True
        if "<|im_end|>" in token_text or "<|endoftext|>" in token_text or "</s>" in token_text:
            return True
        return False
