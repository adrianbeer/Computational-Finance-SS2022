import itertools
perms = list(itertools.permutations([1, 2, 3, 4, 5, 6]))

def check(l):
    i = 0
    after_larger = False
    after_smaller = False
    while l[i] != 4:
        i +=1
    for j in range(i, 6):
        if l[j] > 4: after_larger = True
        if l[j] < 4: after_smaller = True
    if after_larger==True and after_smaller==True:
        print(l)
        return 1

    else:
        return 0

n_perms = 0
for l in perms:
    if check(l): n_perms += 1

print(n_perms)
print(len(perms))