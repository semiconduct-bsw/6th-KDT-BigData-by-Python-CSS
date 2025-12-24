#기본 디폴트
class Animal:
    #생성자
    def __init__(self, type, age):
        self.type = type
        self.age = age

dog = Animal('강아지', 2)
cat = Animal('고양이', 3)

print(dog.type)
print(dog.age)
print(cat.type)
print(cat.age)
