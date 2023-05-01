#%%
import os
dirs_set=[]
for root,dirs,files in os.walk("./Datasets/NELL-995/tasks"):
    for dir in dirs:
        print("dir",dir)
        dirs_set.append(dir)
print(dirs_set)

# %%
a=2
def add(b):
    b=b + 1
    return b
add(a)
print(a)
#%%
a=[1,2,5,3,4,4,None,3,2,None,5]
print("a=",a)
b=list(set(a))
print("b=",b)
print("c=",b.sort(key = a.index))
print(a.remove(None))
# %%
a=1
def c(a):
    a=3

print(a)
c(a)
b=a
print(b)
# %%
with open("test.txt","w") as f:
    f.writelines(str([1,2,3,4]))
# %%
a=[1,2,3,4,5,6]
a1=[11,22,33,45,55,66]
a_index=a.index(4)
b=["a","b","c",4,"d","e"]
b1=["aa","bb","cc","4d","dd","ee"]
b_index=b.index(4)

print(a_index)
print(b_index)

state1=a[:a_index+1]+b[1:b_index][::-1]
state2=b[:b_index+1]+a[1:a_index][::-1]
print("state1:",state1)
print("state2:",state2)
action1=a1[:a_index]+b1[:b_index][::-1]
action2=b1[:b_index]+a1[:a_index][::-1]
print("action1",action1)
print("action2",action2)
#print([11,22,33,'cc', 'bb', 'aa'])
#print([11,22,33,'cc', 'bb', 'aa'][::-1])
#[11, 22, 33, 'cc', 'bb', 'aa']
#['aa', 'bb', 'cc', 33, 22, 11]
# %%
a=[237, 234, 182, 2, 300, 238, 370, 387, 4, 370, 18, 253, 44, 163, 300, 18, 4, 321, 22, 397, 4, 163, 349, 4, 337, 195, 370, 193, 114, 361, 312, 186, 3, 53, 132, 397, 18, 469, 22, 193, 300, 300, 337, 144, 343, 182, 254, 252, 292, 397, 300, 397, 312, 352, 379, 169, 18, 4, 397, 337, 459, 18, 211, 436, 387, 171, 300, 188, 4, 460, 33, 259, 226, 287, 4, 18, 426, 300, 11, 57, 423, 370, 73, 57, 370, 385, 119, 49, 3, 57, 370, 26, 76, 11, 300, 300, 360, 65, 97, 328, 192, 263, 300, 252, 339, 370, 253, 370, 1, 426, 359, 52, 303, 322, 274, 412, 370, 387, 370, 370, 312, 157, 387, 396, 371, 303, 118, 57, 18, 161, 4, 117, 469, 272, 303, 88, 310, 426, 138, 57, 271, 189, 312, 384, 326, 467, 157, 331, 111, 22, 236, 22, 387, 114, 253, 468, 253, 112, 387, 22, 112, 84, 387, 24, 113, 312, 468, 165, 387, 468, 70, 36, 468, 145, 387, 193, 155, 22, 22, 114, 99, 112, 387, 252, 70, 387, 129, 261, 327, 112, 49, 387, 22, 21, 252, 70, 112, 88, 417, 387, 375, 109, 22, 114, 112, 387, 252, 387, 253, 22, 286, 22, 468, 210, 335, 468, 22, 114, 210, 112, 22, 439, 22, 468, 70, 114, 112, 22, 22, 468, 468, 387, 219, 70, 256, 22, 22, 112, 22, 387, 252, 114, 112, 387, 468, 189, 387, 387, 468, 70, 210, 21, 252, 253, 42, 387, 253, 112, 417, 352, 177, 202, 387, 22, 112, 112, 210, 114, 46, 387, 387, 253, 177, 252, 99, 112, 22, 22, 246, 387, 252, 193, 41, 210, 112, 417, 112, 62, 252, 387, 236, 99, 99, 458, 210, 120, 387, 387, 218, 297]
print(len(a))
#%%
from allpairspy import AllPairs
def is_valid_combination(row):
    n = len(row)
    # 设置过滤条件
    if n > 2:
        # 一年级 不能匹配 10-13岁
        if "一年级" == row[1] and "10-13岁" == row[2]:
            return False
    return True
 
parameters = [
    ["男", "女"],
    ["一年级", "二年级", "三年级", "四年级", "五年级"],
    ["8岁以下", "8-10岁", "10-13岁"]
]
 
print("PAIRWISE:")
for i, pairs in enumerate(AllPairs(parameters, filter_func=is_valid_combination)):
    print("用例编号{:2d}: {}".format(i, pairs))

# %%
"""
用例编号 0: Pairs(alpha=0.01, beta=0.001, lr=0.1)
用例编号 1: Pairs(alpha=0.02, beta=0.002, lr=0.1)
用例编号 2: Pairs(alpha=0.05, beta=0.005, lr=0.1)
用例编号 3: Pairs(alpha=0.08, beta=0.008, lr=0.1)
用例编号 4: Pairs(alpha=0.1, beta=0.01, lr=0.1)
用例编号 5: Pairs(alpha=0.1, beta=0.008, lr=0.05)
用例编号 6: Pairs(alpha=0.08, beta=0.005, lr=0.05)
用例编号 7: Pairs(alpha=0.05, beta=0.002, lr=0.05)
用例编号 8: Pairs(alpha=0.02, beta=0.001, lr=0.05)
用例编号 9: Pairs(alpha=0.01, beta=0.01, lr=0.05)
用例编号10: Pairs(alpha=0.01, beta=0.008, lr=0.01)
用例编号11: Pairs(alpha=0.02, beta=0.005, lr=0.01)
用例编号12: Pairs(alpha=0.05, beta=0.01, lr=0.01)
用例编号13: Pairs(alpha=0.08, beta=0.001, lr=0.01)
用例编号14: Pairs(alpha=0.1, beta=0.002, lr=0.01)
用例编号15: Pairs(alpha=0.1, beta=0.001, lr=0.005)
用例编号16: Pairs(alpha=0.08, beta=0.002, lr=0.005)
用例编号17: Pairs(alpha=0.05, beta=0.008, lr=0.005)
用例编号18: Pairs(alpha=0.02, beta=0.01, lr=0.005)
用例编号19: Pairs(alpha=0.01, beta=0.005, lr=0.005)
用例编号20: Pairs(alpha=0.01, beta=0.002, lr=0.001)
用例编号21: Pairs(alpha=0.02, beta=0.008, lr=0.001)
用例编号22: Pairs(alpha=0.05, beta=0.001, lr=0.001)
用例编号23: Pairs(alpha=0.08, beta=0.01, lr=0.001)
用例编号24: Pairs(alpha=0.1, beta=0.005, lr=0.001)

"""