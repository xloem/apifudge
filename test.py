if __name__ == '__main__':
    import apifudge1 as af1
    npapi = af1.API('numpy')
    for name, text in npapi.get_examples().items():
        print(f'{name}: {repr(text)}')
    npapi.set_example(
'''Construct an array B that repeats array Y 4 times along the last dimension.''',
'''B = np.tile(Y, 4)''')
    for name, text in npapi.get_examples().items():
        print(f'{name}: {repr(text)}')
