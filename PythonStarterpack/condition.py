accuracy = 0.82 #예시로 0.95일 때는 90% 이상이 출력될 것
if accuracy > 0.9:
    #로직
    print('90% 이상')
elif accuracy > 0.8:
    print('80% 이상')
elif accuracy > 0.7:
    print('70% 이상')
else:
    print('70% 미만')
