class Person:
    def __call__(self, name):
        print("__call__" + "hello" + name)

    @staticmethod
    def hello(name):
        print("hello" + name)


person = Person()
person('zzy')
person.hello('lisi')

