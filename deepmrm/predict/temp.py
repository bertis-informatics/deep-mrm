a = None

def test(b):
    global a
    a = b

def print_a():
    global a
    print(a)