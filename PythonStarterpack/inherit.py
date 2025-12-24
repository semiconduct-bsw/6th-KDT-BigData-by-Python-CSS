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
