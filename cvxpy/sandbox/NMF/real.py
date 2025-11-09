import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp

np.random.seed(0)

def make_circle(n=20, r=5):
    y, x = np.ogrid[-n//2:n//2, -n//2:n//2]
    mask = x**2 + y**2 <= r**2
    return mask.astype(float)

def make_square(n=20, s=10):
    img = np.zeros((n, n))
    start = (n - s)//2
    img[start:start+s, start:start+s] = 1
    return img

def make_triangle(n=20, s=12):
    img = np.zeros((n, n))
    for i in range(s):
        row = n//2 + i - s//2
        if 0 <= row < n:
            start = max(0, n//2 - i//2)
            end = min(n, n//2 + i//2)
            img[row, start:end] = 1
    return img

# base shapes
bases = [make_circle(), make_square(), make_triangle()]
n_bases = len(bases)


# create 100 random images as mixtures of the bases plus noise
n_samples = 100
true_images = []
noises = []
for _ in range(n_samples):
    coeffs = 10 * np.random.rand(n_bases)
    coeffs /= coeffs.sum()  # normalize to sum to 1
    true_img = sum(c * b for c, b in zip(coeffs, bases))
    noise = 0.2 * np.random.randn(*true_img.shape)
    # clip noise to avoid negative values
    noise[true_img + noise < 0] = -true_img[true_img + noise < 0]
    true_images.append(true_img.flatten())
    noises.append(noise.flatten())


A_true = np.array(true_images) 
noise_array = np.array(noises)
A = A_true + noise_array
assert(A.min() >= 0), "Data matrix contains negative values!"

n = A.shape[0]
m = A.shape[1]
k = 3  # number of components
X = cp.Variable((n, k), bounds=[0, None])
Y = cp.Variable((k, m), bounds=[0, None])
X.value = np.random.rand(n, k)
Y.value = np.random.rand(k, m)
obj = cp.sum(cp.square(A - X @ Y))
problem = cp.Problem(cp.Minimize(obj))
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True, derivative_test='none')
X = X.value
Y = Y.value

# apply nmf
#k = 3  
#model = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=500)
#X = model.fit_transform(A)
#Y = model.components_


# Denoised reconstruction
A_denoised = X @ Y


# -------------------------------------------------------------------------------------
#                               plot results
# -------------------------------------------------------------------------------------
fig, axes = plt.subplots(4, 6, figsize=(14, 10))
plt.tight_layout()
fig.subplots_adjust(hspace=0.10, top=0.95)

for i in range(3):
    axes[0, i].imshow(bases[i], cmap="gray")
    axes[0, i].axis("off")


index = [0, 2, 1]
for j in [0, 1, 2]:
    axes[0, 3 + j].imshow(Y[index[j]].reshape(20, 20), cmap="gray")
    axes[0, 3 + j].axis("off")

fig.text(0.25, 0.96, "True basis images", ha="center", va="center",
         fontsize=22, fontweight="bold")

fig.text(0.75, 0.96, "Recovered basis images", ha="center", va="center",
         fontsize=22, fontweight="bold")

fig.text(0.5, 0.72, "Original images", ha="center", va="center",
         fontsize=22, fontweight="bold")

fig.text(0.5, 0.49, "Noisy images", ha="center", va="center",
         fontsize=22, fontweight="bold")

fig.text(0.5, 0.26, "Denoised images", ha="center", va="center",
         fontsize=22, fontweight="bold")

# true images
for i in range(6):
    axes[1, i].imshow(A_true[i].reshape(20, 20), cmap="gray")
    axes[1, i].axis("off")

# noisy images
for i in range(6):
    axes[2, i].imshow(A[i].reshape(20, 20), cmap="gray")
    axes[2, i].axis("off")

# denoised images
for i in range(6):
    axes[3, i].imshow(A_denoised[i].reshape(20, 20), cmap="gray")
    axes[3, i].axis("off")

plt.savefig("NMF.pdf")













# visualize
#fig, axes = plt.subplots(3, 6, figsize=(10, 5))
#
## (a) True shapes
#for i, b in enumerate(bases):
#    axes[0, i].imshow(b, cmap="gray")
#    axes[0, i].set_title(f"true shape {i+1}")
#    axes[0, i].axis("off")
#
#
#
#
## Fill the rest of the row
#for j in range(3, 6):
#    axes[0, j].axis("off")
#
## (b) Random original mixtures
#for i in range(6):
#    axes[1, i].imshow(A[i].reshape(20, 20), cmap="gray")
#    axes[1, i].set_title(f"Mixture {i+1}")
#    axes[1, i].axis("off")
#
## (c) Learned NMF components
#for i in range(k):
#    axes[2, i].imshow(Y[i].reshape(20, 20), cmap="gray")
#    axes[2, i].set_title(f"NMF comp {i+1}")
#    axes[2, i].axis("off")
#
#for j in range(k, 6):
#    axes[2, j].axis("off")
#
## Denoised reconstruction
#A_denoised = X @ Y
#
#plt.tight_layout()
#plt.savefig("nmf_real_test.pdf")
#
## Compare original vs. denoised
#fig, axes = plt.subplots(3, 6, figsize=(10, 4))
#for i in range(6):
#    axes[0, i].imshow(A[i].reshape(20, 20), cmap='gray')
#    axes[0, i].set_title("Noisy")
#    axes[0, i].axis("off")
#
#    axes[1, i].imshow(A_denoised[i].reshape(20, 20), cmap='gray')
#    axes[1, i].set_title("Denoised (NMF)")
#    axes[1, i].axis("off")
#
#    axes[2, i].imshow(A_true[i].reshape(20, 20), cmap='gray')
#    axes[2, i].set_title("True")
#    axes[2, i].axis("off")
#
#plt.tight_layout()
#plt.savefig("nmf_denoising_comparison.pdf")
#