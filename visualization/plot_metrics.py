import matplotlib.pyplot as plt


def plot_ber(ber_w, ber_s):

    plt.figure(figsize=(5, 5))

    plt.bar(["Weak User", "Strong User"],
            [ber_w, ber_s])

    plt.ylabel("BER")
    plt.title("Bit Error Rate")

    plt.show()