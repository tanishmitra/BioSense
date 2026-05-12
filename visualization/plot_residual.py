import matplotlib.pyplot as plt


class ResidualPlotter:

    def plot(self, env, spikes=None):

        plt.figure(figsize=(10, 4))

        plt.plot(env)

        if spikes is not None and len(spikes) > 0:
            plt.scatter(spikes, env[spikes])

        plt.title("SIC Residual Envelope")
        plt.xlabel("Samples")
        plt.ylabel("Residual Magnitude")

        plt.grid(True)
        plt.show()