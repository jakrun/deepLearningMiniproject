import csv
import matplotlib.pyplot as plt

epochs, train_loss, val_loss, val_acc = [], [], [], []
with open("metrics.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        val_loss.append(float(row["val_loss"]))
        val_acc.append(float(row["val_acc"]))

best_epoch = max(range(len(val_acc)), key=lambda i: val_acc[i])
best_acc = val_acc[best_epoch]
print(f"Best val accuracy: {best_acc * 100:.2f}% at epoch {epochs[best_epoch]}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(epochs, train_loss, label="train loss", color="tab:blue")
ax1.plot(epochs, val_loss, label="val loss", color="tab:orange")
ax1.axvline(epochs[best_epoch], color="grey", linestyle="--", linewidth=0.8,
            label=f"best epoch ({epochs[best_epoch]})")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training and validation loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, [a * 100 for a in val_acc], color="tab:green", label="val accuracy")
ax2.axhline(best_acc * 100, color="grey", linestyle=":", linewidth=0.8,
            label=f"best = {best_acc * 100:.2f}%")
ax2.axvline(epochs[best_epoch], color="grey", linestyle="--", linewidth=0.8)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation accuracy (%)")
ax2.set_title("Validation accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150, bbox_inches="tight")
print("Saved loss_curves.png")
plt.show()
