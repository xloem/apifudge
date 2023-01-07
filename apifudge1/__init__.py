import transformers, torch, accelerate
import itertools, os, random

# let's make an interface to collect user examples for apis
    # these are scrapable, but let's not scrape them yet.

# then we can collect those examples in a prompt for a simple generator.

class API:
    def __init__(self, name, HOME = os.path.expanduser('~/.apifudge')):
        self.name = name
        self.path = os.path.join(HOME, self.name)
        self.examples_path = os.path.join(self.path, 'examples')
        os.makedirs(self.examples_path, exist_ok=True)

    def get_examples(self):
        return {
            prompt: result
            for fn in os.listdir(self.examples_path)
            for prompt, result in (open(os.path.join(self.examples_path, fn)).read().split('\n', 1),)
        }
    def set_example(self, prompt, result, *params, name=None, **kwparams):
        if name is None:
            name = ''.join([
                char
                for char in prompt.replace(' ','_').lower()
                if char in 'abcdefghijklmnopqrstuvwxyz0123456789_-'
            ]) + '.txt'
        fn = os.path.join(self.examples_path, name)
        try:
            with open(fn + '.new', 'w') as f:
                f.write(prompt + '\n')
                f.write(result)
            os.rename(fn + '.new', fn)
        except:
            os.unlink(fn + '.new')
            raise

class Generator:
    # note: i had reasonable results using pad_token_id for sep,
    # but it requires a little finesse when sampling multiple values, or generating middle terminators: it doesn't generate the pad token
    def __init__(self, pipeline = None, sep_token = '\n', eos_token = None):
        if pipeline is None:
            pipeline = transformers.pipeline(
                    'text-generation',
                    'bigscience/bloomz-560m',
                    model_kwargs=dict(device_map='auto')
                )
        self.model = pipeline.model
        self.tokenizer = pipeline.tokenizer
        self.sep_token_ids = self.tokenizer.encode(sep_token)
        self.eos_token_ids = self.tokenizer.encode(eos_token) if eos_token else [self.tokenizer.eos_token_id]
        # disallow zero-length generations
        self.model.config.begin_suppress_tokens = [self.sep_token_ids[0], self.eos_token_ids[0]]
    @property
    def device(self):
        return self.model.device
    def _encode_one(self, ex_prompt = None, ex_result = None):
        token_ids = []
        token_ids.append(self.tokenizer.bos_token_id)
        if ex_prompt is not None:
            if type(ex_prompt) is str:
                ex_prompt = self.tokenizer.encode(ex_prompt)
            if ex_prompt[-1] == self.tokenizer.eos_token_id:
                ex_prompt = ex_prompt[:-1]
            assert ex_prompt[-1] not in self.model.config.begin_suppress_tokens
            token_ids.extend(ex_prompt)
            token_ids.extend(self.sep_token_ids)
            if ex_result is not None:
                if type(ex_result) is str:
                    ex_result = self.tokenizer.encode(ex_result)
                if ex_result[-1] == self.tokenizer.eos_token_id:
                    ex_result = ex_result[:-1]
                assert ex_prompt[-1] not in self.model.config.begin_suppress_tokens
                token_ids.extend(ex_result)
                token_ids.extend(self.eos_token_ids)
        else:
            assert ex_result is None
        return token_ids
    def _forward(self, tokens_ids, **kwparams):
        if type(tokens_ids[0]) is int:
            tokens_ids = [tokens_ids]
        max_len = max([len(token_ids) for token_ids in tokens_ids])
        attn_mask = torch.tensor([
                [0] * (max_len - len(token_ids)) + [1] * len(token_ids)
                for token_ids in tokens_ids
            ], device=self.device)
        tokens_ids = torch.tensor([
                [self.tokenizer.pad_token_id] * (max_len - len(token_ids)) + token_ids
                for token_ids in tokens_ids
            ], device=self.device)
        outputs_ids = self.model.generate(
                tokens_ids,
                attention_mask=attn_mask,
                max_length=kwparams.pop('max_length', self.model.config.seq_length),
                **kwparams
            )
        outputs_ids = outputs_ids[...,tokens_ids.shape[-1]:]
        assert (outputs_ids >= 0).all()
        outputs_ids = [
            (output_ids[:torch.where(output_ids == self.tokenizer.pad_token_id)[0][0]] 
             if output_ids[-1] == self.tokenizer.pad_token_id else output_ids)
            for output_ids in outputs_ids
        ]
        if len(outputs_ids) == 1:
            outputs_ids = outputs_ids[0]
        return outputs_ids
    def __call__(self, api, prompt):
        token_ids = []
        for ex_prompt, ex_result in api.get_examples().items():
            token_ids.extend(self._encode_one(ex_prompt, ex_result))
        token_ids.extend(self._encode_one(prompt))
        output_ids = self._forward(
            token_ids,
        )
        return self.tokenizer.decode(output_ids, skip_special_tokens = True)
    def augment(self, api, count=8):
        examples_ids = []
        for ex_prompt, ex_result in api.get_examples().items():
            examples_ids.append(self._encode_one(ex_prompt, ex_result))
        random.shuffle(examples_ids)
        token_ids = list(itertools.chain.from_iterable(examples_ids))
        # i looked at hf beam search, and decided not to use it because it required to upload all the hypothesis beams
        # to the gpu together, which means arbitrarily limiting the width of the tree exploration to a small value
        # i began implementing a beam search, but was not focused enough to do it correctly
        # i have a disparate working implementation in adventure3 that could be copied
        # but the change to huggingface transformers might not be that significant
        #    prompts_ids = simple_beam_search(self.model, token_ids + self._encode_one(), count, tokenizer=self.tokenizer)
        # sampling seems to work well enough for now
        prompts_ids = self._forward(token_ids + self._encode_one(), num_return_sequences=count, do_sample=True)
        results_ids = self._forward([token_ids + self._encode_one(prompt_ids) for prompt_ids in prompts_ids])
        return list(zip(self.tokenizer.batch_decode(prompts_ids, skip_special_tokens = True), self.tokenizer.batch_decode(results_ids, skip_special_tokens = True)))
