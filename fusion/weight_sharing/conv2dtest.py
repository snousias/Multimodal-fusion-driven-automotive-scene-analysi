import torch
import numpy as np
import time
torch.manual_seed(0)

input = torch.randn(1,12, 64, 64)
m = torch.nn.Conv2d(12, 64, 3, stride=1,bias=False)
output = m(input)




INPUT=input.numpy()
M=m.weight.cpu().detach().numpy()
OUTPUT=output.cpu().detach().numpy()

DIFF=1


# print('Depthwize 3D Convolution')
# sttime=time.time()
# OUTPUT_T1=np.empty((np.shape(INPUT)[0],np.shape(M)[0],np.shape(INPUT)[2]-2,np.shape(INPUT)[3]-2))
# for i in range(np.shape(M)[0]):
#     K=M[i]
#     for x in range(1,np.shape(INPUT)[2]-1):
#         for y in range(1,np.shape(INPUT)[3]-1):
#             S=INPUT[0,:,x-1:x+2,y-1:y+2]
#             U=np.sum(np.multiply(K, S))
#             OUTPUT_T1[0,i,x-1,y-1]=U
# print("Diff to original:")
# DIFF=np.sum(OUTPUT-OUTPUT_T1)
# print(DIFF)
# print('ok')
# print("Time:")
# print(time.time()-sttime)
# print(20*"=")
#
# print('Depthwize 3D Convolution with dot products')
# sttime=time.time()
# OUTPUT_T2=np.empty((np.shape(INPUT)[0],np.shape(M)[0],np.shape(INPUT)[2]-2,np.shape(INPUT)[3]-2))
# for i in range(np.shape(M)[0]):
#     K=M[i]
#     for x in range(1,np.shape(INPUT)[2]-1):
#         for y in range(1,np.shape(INPUT)[3]-1):
#             S=INPUT[0,:,x-1:x+2,y-1:y+2]
#             ACCUMTEMP=np.empty((np.shape(K)[1],np.shape(K)[2]))
#             for xs in range(np.shape(K)[1]):
#                 for ys in range(np.shape(K)[2]):
#                     K_i=K[:,xs,ys]
#                     S_i=S[:,xs,ys]
#                     U_i=np.dot(K_i,S_i)
#                     ACCUMTEMP[xs,ys]=U_i
#             OUTPUT_T2[0,i,x-1,y-1]=np.sum(ACCUMTEMP)
# print("Diff to original:")
# DIFF=np.sum(OUTPUT-OUTPUT_T2)
# print(DIFF)
# print('ok')
# print("Time:")
# print(time.time()-sttime)
# print(20*"=")




sharing=True
sharingdict=[
    ((0, 0), (0, 0)),
    ((0, 1), (0, 1)),
    ((0, 2), (0, 0)),
    ((1, 0), (0, 1)),
    ((1, 1), (0, 0)),
    ((1, 2), (0, 1)),
    ((2, 0), (0, 0)),
    ((2, 1), (0, 1)),
    ((2, 2), (0, 0))
]

# sharingdict=[
#     ((0, 0), (0, 0)),
#     ((0, 1), (0, 1)),
#     ((0, 2), (0, 2)),
#     ((1, 0), (1, 0)),
#     ((1, 1), (1, 1)),
#     ((1, 2), (1, 2)),
#     ((2, 0), (2, 0)),
#     ((2, 1), (2, 1)),
#     ((2, 2), (2, 2))
# ]


Source = [row[0] for row in sharingdict]
Target = [row[1] for row in sharingdict]
ToBeUsedInstead = []
for x in Target:
    if x not in ToBeUsedInstead:
        ToBeUsedInstead.append(x)
sharingdictsolver=[]
for x in ToBeUsedInstead:
    sharingdictsolver.append((x,[]))
    # print("OJK")
for x in sharingdictsolver:
    for i,y in enumerate(Target):
        if x[0]==y:
            x[1].append(Source[i])





###########################################################################
print('Multiply I')
sttime=time.time()
matmultime=0
zshape = np.shape(INPUT)[1]
xshape = np.shape(INPUT)[2]
yshape = np.shape(INPUT)[3]
Mask=np.zeros((zshape, xshape - 2, yshape - 2))
for scanfactor in range(2):
    for xinput in range(1, xshape - 1):
        Mask[:, xinput - 1, 0+scanfactor] = np.ones(zshape)
        Mask[:, xinput - 1, yshape - 3 - scanfactor] = np.ones(zshape)
    for yinput in range(1, yshape - 1):
        Mask[:, 0 + scanfactor, yinput - 1] = np.ones(zshape)
        Mask[:, xshape - 3 - scanfactor, yinput - 1] = np.ones(zshape)
OUTPUT_T4 = np.empty((np.shape(INPUT)[0], np.shape(M)[0], xshape - 2, yshape - 2))
for i in range(np.shape(M)[0]):
    K=M[i]
    if sharing:
        for row in sharingdict:
            xs_init = row[0][0]
            ys_init = row[0][1]
            xs_targ = row[1][0]
            ys_targ = row[1][1]
            K[:, xs_init, ys_init] = K[:, xs_targ, ys_targ]

    matmultimestart = time.time()
    zshape = np.shape(INPUT)[1]
    xshape = np.shape(INPUT)[2]
    yshape = np.shape(INPUT)[3]
    ACCUMTEMP = np.zeros((zshape, xshape - 2, yshape - 2))
    for xs in range(np.shape(K)[1]):
        for ys in range(np.shape(K)[2]):
            I_part = INPUT[0, :, xs:xs + xshape - 2, ys:ys + yshape - 2]
            K_i = K[:, xs, ys]
            K_rep = K_i.reshape((np.shape(K_i)[0], 1, 1)).repeat(xshape - 2, 1).repeat(yshape - 2, 2)
            UMM = np.multiply(K_rep, I_part)
            # UMM=np.ma.multiply(K_rep, I_part,mask=Mask)
            ACCUMTEMP = ACCUMTEMP + UMM
    matmultime=matmultime+(time.time()-matmultimestart)
    OUTPUT_T4[0, i, :, :] = np.sum(ACCUMTEMP, axis=0)
print("Diff to original:")
DIFF=np.sum(OUTPUT-OUTPUT_T4)
print(DIFF)
print("Time:")
print(time.time()-sttime)
print(20*"-")

print("Matmul time:")
print(matmultime)
print(20*"=")





#######################################################################
print('Multiply II')
matmultime=0
sttime=time.time()
zshape = np.shape(INPUT)[1]
xshape = np.shape(INPUT)[2]
yshape = np.shape(INPUT)[3]
OUTPUT_T5 = np.empty((np.shape(INPUT)[0], np.shape(M)[0], xshape - 2, yshape - 2))
for i in range(np.shape(M)[0]):
    K = M[i]
    if sharing:
        for row in sharingdict:
            xs_init = row[0][0]
            ys_init = row[0][1]
            xs_targ = row[1][0]
            ys_targ = row[1][1]
            K[:, xs_init, ys_init] = K[:, xs_targ, ys_targ]

    matmultimestart = time.time()
    ACCUMTEMP = np.zeros((zshape, xshape - 2, yshape - 2))
    I_M = INPUT[0, :, 2:xshape - 2, 2:yshape - 2]
    for row in sharingdict:
        xs_init = row[0][0]
        ys_init = row[0][1]
        xs_targ = row[1][0]
        ys_targ = row[1][1]
        xs=xs_init
        ys=ys_init
        K_i = K[:, xs, ys]
        K_M = K_i.reshape((np.shape(K_i)[0], 1, 1)).repeat(np.shape(I_M)[1], 1).repeat(np.shape(I_M)[1],2)
        UMMtemp_M=np.multiply(I_M,K_M)
        xs_opposite = 2 - xs
        ys_opposite = 2 - ys
        ACCUMTEMP[:, xs_opposite:xs_opposite + np.shape(UMMtemp_M)[1],
        ys_opposite:ys_opposite + np.shape(UMMtemp_M)[2]] = \
            ACCUMTEMP[:, xs_opposite:xs_opposite + np.shape(UMMtemp_M)[1],
            ys_opposite:ys_opposite + np.shape(UMMtemp_M)[2]] + UMMtemp_M
    matmultime = matmultime + (time.time() - matmultimestart)
    OUTPUT_T5[0, i, :, :] = np.sum(ACCUMTEMP, axis=0)




    # Xsss
    for scanfactor in range(2):
        for xinput in range(1, xshape - 1):
            InputVolume = INPUT[0, :, xinput - 1:xinput + 2, 0+scanfactor:3+scanfactor]
            Ain = np.sum(np.multiply(K, InputVolume))
            OUTPUT_T5[0, i, xinput - 1, 0+scanfactor] = Ain

            InputVolume = INPUT[0, :, xinput - 1:xinput + 2, yshape - 3-scanfactor:yshape-scanfactor]
            Ain = np.sum(np.multiply(K, InputVolume))
            OUTPUT_T5[0, i, xinput - 1, yshape - 3-scanfactor] = Ain

        # Ysss
        for yinput in range(1, yshape - 1):
            InputVolume = INPUT[0, :, 0+scanfactor:3+scanfactor, yinput - 1:yinput + 2]
            Ain = np.sum(np.multiply(K, InputVolume))
            OUTPUT_T5[0, i, 0+scanfactor, yinput - 1] = Ain

            InputVolume = INPUT[0, :, xshape - 3-scanfactor:xshape-scanfactor, yinput - 1:yinput + 2, ]
            Ain = np.sum(np.multiply(K, InputVolume))
            OUTPUT_T5[0, i, xshape - 3-scanfactor, yinput - 1] = Ain
print("Diff to original:")
DIFF = np.sum(OUTPUT - OUTPUT_T5)
print(DIFF)
print("Time:")
print(time.time()-sttime)
print(20*"-")
print("Matmul time:")
print(matmultime)
print(20*"=")

#######################################################################
print('Sharing')
matmultime=0
sttime=time.time()
zshape = np.shape(INPUT)[1]
xshape = np.shape(INPUT)[2]
yshape = np.shape(INPUT)[3]
OUTPUT_T6 = np.empty((np.shape(INPUT)[0], np.shape(M)[0], xshape - 2, yshape - 2))
for i in range(np.shape(M)[0]):
    K = M[i]
    if sharing:
        for row in sharingdict:
            xs_init = row[0][0]
            ys_init = row[0][1]
            xs_targ = row[1][0]
            ys_targ = row[1][1]
            K[:, xs_init, ys_init] = K[:, xs_targ, ys_targ]

    matmultimestart = time.time()
    ACCUMTEMP = np.zeros((zshape, xshape - 2, yshape - 2))
    I_M = INPUT[0, :, 2:xshape - 2, 2:yshape - 2]
    for row in sharingdictsolver:
        K_i = K[:, row[0][0], row[0][1]]
        K_M = K_i.reshape((np.shape(K_i)[0], 1, 1)).repeat(np.shape(I_M)[1], 1).repeat(np.shape(I_M)[1], 2)
        UMMtemp_M = np.multiply(I_M, K_M)
        for g in row[1]:
            xs=g[0]
            ys=g[1]
            xs_opposite = 2 - xs
            ys_opposite = 2 - ys
            ACCUMTEMP[:, xs_opposite:xs_opposite + np.shape(UMMtemp_M)[1],
            ys_opposite:ys_opposite + np.shape(UMMtemp_M)[2]] = \
                ACCUMTEMP[:, xs_opposite:xs_opposite + np.shape(UMMtemp_M)[1],
                ys_opposite:ys_opposite + np.shape(UMMtemp_M)[2]] + UMMtemp_M
    matmultime = matmultime + (time.time() - matmultimestart)
    OUTPUT_T6[0, i, :, :] = np.sum(ACCUMTEMP, axis=0)

    # Xsss
    for scanfactor in range(2):
        for xinput in range(1, xshape - 1):
            InputVolume = INPUT[0, :, xinput - 1:xinput + 2, 0 + scanfactor:3 + scanfactor]
            Ain = np.sum(np.multiply(K, InputVolume))
            OUTPUT_T6[0, i, xinput - 1, 0 + scanfactor] = Ain

            InputVolume = INPUT[0, :, xinput - 1:xinput + 2, yshape - 3 - scanfactor:yshape - scanfactor]
            Ain = np.sum(np.multiply(K, InputVolume))
            OUTPUT_T6[0, i, xinput - 1, yshape - 3 - scanfactor] = Ain

        # Ysss
        for yinput in range(1, yshape - 1):
            InputVolume = INPUT[0, :, 0 + scanfactor:3 + scanfactor, yinput - 1:yinput + 2]
            Ain = np.sum(np.multiply(K, InputVolume))
            OUTPUT_T6[0, i, 0 + scanfactor, yinput - 1] = Ain

            InputVolume = INPUT[0, :, xshape - 3 - scanfactor:xshape - scanfactor, yinput - 1:yinput + 2, ]
            Ain = np.sum(np.multiply(K, InputVolume))
            OUTPUT_T6[0, i, xshape - 3 - scanfactor, yinput - 1] = Ain
print("Diff to original:")
DIFF = np.sum(OUTPUT - OUTPUT_T6)
print(DIFF)
print("Time:")
print(time.time()-sttime)
print(20*"-")
print("Matmul time:")
print(matmultime)
print(20*"=")
########################################################################3