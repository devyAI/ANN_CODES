import numpy as np

class McCulloh_Pitt:
    def __init__(self, num_inputs):
        self.weights = np.zeros(num_inputs)
        self.threshold = 0

    def set_weights(self, weights):
        if len(weights) != len(self.weights):
            raise ValueError("Number of weights must match number of inputs")
        self.weights = np.array(weights)

    def set_threshold(self,threshold):
        self.threshold = threshold

    def activation(self, net_input):
        return 1 if net_input>self.threshold else 0

    def forward_pass(self, inputs):
        net_input = np.dot(inputs, self.weights)
        return self.activation(net_input)
    
def generate_ANDNOT():
    mp = McCulloh_Pitt(2)
    mp.set_weights([1,-1])
    mp.set_threshold(0)

    truth_table = [(0,0), (0,1), (1,0), (1,1)]

    for inputs in truth_table:
        output = mp.forward_pass(inputs)
        print(f"{inputs[0]}\t{inputs[1]}\t{output}")

def main():
    while True:
        print("Menu:")
        print("1. Generate ANDNOT using mp neuron")
        print("2.exit")

        choice = int(input("enter choice"))
        if choice == 1:
            generate_ANDNOT()
        elif choice == 2:
            print("exiting")
            break
        else:
            print("invalid")

if __name__ == "__main__":
    main()




