if __name__ == '__main__':
    import apifudge1 as af1
    npapi = af1.API('numpy')
    npapi.set_example(
'''Construct an array B that repeats array Y 4 times along the last dimension.''',
'''B = np.tile(Y, 4)''')
    npapi.set_example(
'''Swap the last two axes of an array D.''',
'''D = D.swapaxes(-2, -1)''')
    npapi.set_example(
'''Sort an array along the third axis.''',
'''array.sort(3)''')
    for name, text in npapi.get_examples().items():
        print(f'{name}: {repr(text)}')
    print('Inferring:')
    prompt = 'Reverse an array C.'
    output = generator(npapi, prompt)
    print(f'{prompt}: {repr(output)}')
    generator = af1.Generator()
    print('Augmenting ...')
    for prompt, result in generator.augment(npapi):
        print(f'{prompt}: {repr(result)}')
