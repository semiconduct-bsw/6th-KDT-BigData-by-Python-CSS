class Animal:
    #생성자
    def __init__(self, type, age, name):
        self.type = type
        self.age = age
        self.name = name
    
    #함수
    def bark(self):
        print(f'{self.name}이/가 짖습니다.')

dog = Animal('강아지', 2, '어엉')
cat = Animal('고양이', 3, '웅어')

print(dog.type)
print(dog.age)
print(dog.name)

dog.bark()
cat.bark()
