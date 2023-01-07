if __name__ == '__main__':
    import apifudge1 as af1
    npapi = af1('numpy')
    for name, text in npapi.get_examples().items():
        print(f'{name}: {repr(text)}')
    npapi.set_example(
'''Repeat the array Y
''')
    
