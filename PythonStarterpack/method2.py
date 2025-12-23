def add_two_numbers(num1, num2) :
    sum = num1 + num2

# 함수 호출
result = add_two_numbers(10, 20)
print(result)

------------------

def add_two_numbers(num1, num2) :
    sum = num1 + num2
    return sum

# 함수 호출
result = add_two_numbers(10, 20)
print(result)

#퀴즈
one = get_element_in_list(['사과', '바나나'], 1)
print(one) # 이것이 바나나가 되도록 해보기

# 퀴즈 혼자 풀이
def get_element_list(ver1, ver2, numA) :
    list1 = [ver1, ver2]
    list2 = list1[numA]
    return list2

one = get_element_list('사과','바나나',1)
print(one)

#퀴즈 해답
def get_element_in_list(list,index):
   element = list[index]
   return element

one = get_element_in_list(['사과','바나나'],1)
print(one)

one1 = get_element_in_list(['a','b','c'],2)
print(one1)
