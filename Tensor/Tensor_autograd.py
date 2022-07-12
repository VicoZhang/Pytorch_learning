#  彰显 autograd 的魅力

import torch

print("*--------------------------------------------------*")
a = torch.arange(9.).view(3, 3)
print("a={}\na的autograd属性为:{}".format(a, a.requires_grad))
print("*--------------------------------------------------*")
print("打开a的autograd属性")
a.requires_grad = True
print("a={}".format(a))
print("*--------------------------------------------------*")
print("由a计算out = mean((a[i]+1)^2)")
out = torch.mean((a+1)**2)
print("out={}".format(out))
print("*--------------------------------------------------*")
print("发现out存在属性: out.grad_fn = {}".format(out.grad_fn))
print("说明out存储了由a计算得出out的途径")
print("*--------------------------------------------------*")
print("调用backward()对out自动计算由a到out的微分：d(out)/d(a)")
out.backward()
print("可以得出a的微分值: a.grad={}".format(a.grad))
print("*--------------------------------------------------*")
print("验算")
print("out对a的微分 d(out)/d(a) = 2(a[i]+1)/9")
c = 2*(a+1)/9
print(c)
print("即可验证a的autograd属性")

# 矩阵的求导
# x = torch.arange(9., requires_grad=True).view(3, 3)
# w = torch.ones(3, 3, requires_grad=True)
# A = torch.mm(w, x)
# print("A={}".format(A))
# result = torch.mean(A)
# print("result={}".format(result))
# result.backward()
# print(w.grad)
