class Human:
    def __init__(self, country, gender):
        self.country = country
        self.gender = gender

# Human 내용을 Korean으로 땡기고 싶음
class Korean(Human):
    def __init__(self, name, age, gender):
        super().__init__('대한민국', gender)
        self.name = name
        self.age = age

class American(Human):
    def __init__(self, name, age, gender):
        super().__init__('미국', gender)
        self.name = name
        self.age = age

korean = Korean('홍길동', 20, '남자')
american = American('John Doe', 25, '여자')

print(korean.country)
print(korean.name)
print(korean.age)
