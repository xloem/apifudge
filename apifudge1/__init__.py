import transformers
import os

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
