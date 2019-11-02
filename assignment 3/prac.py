t=int(input())
while t>0:
    n,m,k,l = map(int,input().split())
    arr = list(map(int,input().split())) 
    di=dict()
    ti = 0
    for ind in arr:
        if ind not in di:
            di[ind]=1
        else:
            di[ind]+=1
            ti=0
    for i in range(k):
        if i==0:
            continue
        if i%l==0 and i>0:
            m-=1
        if i==1:
            ti=(l-(i%l))+(m*l)
        else:
            ti2=(l-(i%l))+(m*l)
            if(ti2<ti):
                ti=ti2
        if i in di:
            m+=di[i]
         
    print(ti)
    t -= 1
            
            
        