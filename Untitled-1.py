dic = {'1':'一','2':'二','3':'三','4':'四','5':'五','6':'六','7':'七','8':'八','9':'九','0':'零'}
s = input()
p = []
for i in range(len(s)):
    p.append(dic[s[i]])
answer = ''.join(p)
print(answer)