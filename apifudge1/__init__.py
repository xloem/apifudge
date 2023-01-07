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
    def __init__(self, pipeline = None):
        if pipeline is None:
            pipeline = transformers.pipeline(
                    'text-generation',
                    'bigscience/bloomz-560m',
                    model_kwargs=dict(device_map='auto')
                )
        self.model = pipeline.model
        self.tokenizer = pipeline.tokenizer
        # pad token disabled since it may be used in the prompt
        self.model.config.pad_token_id = -1
        # disallow zero-length generations
        self.model.config.begin_suppress_tokens = [self.tokenizer.eos_token_id]
    @property
    def device(self):
        return self.model.device
    def _encode_one(self, ex_prompt = None, ex_result = None):
        token_ids = []
        token_ids.append(self.tokenizer.bos_token_id)
        if ex_prompt is not None:
            if type(ex_prompt) is str:
                ex_prompt = self.tokenizer.encode(ex_prompt)
            token_ids.extend(ex_prompt)
            token_ids.append(self.tokenizer.pad_token_id)
            if ex_result is not None:
                if type(ex_result) is str:
                    ex_result = self.tokenizer.encode(ex_result)
                token_ids.extend(ex_result)
                token_ids.append(self.tokenizer.eos_token_id)
        else:
            assert ex_result is None
        return token_ids
    def _forward(self, token_ids, **kwparams):
        token_ids = torch.tensor(token_ids, device=self.device)[None,:]
        output_ids = self.model.generate(token_ids, attention_mask=torch.ones_like(token_ids), max_length=self.model.config.seq_length, **kwparams)
        output_ids = output_ids[0]
        output_ids = output_ids[token_ids.shape[-1]:(-1 if output_ids[-1] in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id) else None)]
        return output_ids
    def __call__(self, api, prompt):
        token_ids = []
        for ex_prompt, ex_result in api.get_examples().items():
            token_ids.extend(self._encode_one(ex_prompt, ex_result))
        token_ids.extend(self._encode_one(prompt))
        output_ids = self._forward(
            token_ids,
        )
        return self.tokenizer.decode(output_ids)
    def augment(self, api):
        # would be much more efficient to explore the first generation possibilities, rather than continuing to accumulate
        examples_ids = []
        for ex_prompt, ex_result in api.get_examples().items():
            examples_ids.append(self._encode_one(ex_prompt, ex_result))
        while True:
            random.shuffle(examples_ids)
            token_ids = list(itertools.chain.from_iterable(examples_ids))
            prompt_ids = self._forward(token_ids + self._encode_one())
            result_ids = self._forward(token_ids + self._encode_one(prompt_ids))
            if len(token_ids) + len(prompt_ids) + len(result_ids) + 3 >= self.model.config.seq_length:
                break
            yield self.tokenizer.decode(prompt_ids), self.tokenizer.decode(result_ids)
            examples_ids.append(self._encode_one(prompt_ids, result_ids))
