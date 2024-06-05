import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from keras.models import Sequential 
from keras.layers import Dense #Fully connected layer in a NN - each neuron is connected to every neuron in previous and next layer, we can specify number of neurons in layer, activation function and other parameters
from keras.optimizers import Adam 

# Define Constants
TRAINING_SPLIT = 0.8 #0.8/0.2 split based on recommended practice
BIOMASS_FILE = "biomass.txt" 
COMPOUND_FILE = "compoundA.txt"
SUBSTRATE_FILE = "substrate.txt"

# Global Training Parameters:
LOSS_FUNCTION = 'mean_squared_error' # Common loss function for regression problem - better because it summarises the difference between continuous variable better than other functions
LEARNING_RATE = 0.0035 # Learning rate varies the update rate for the weights. Lower learning rate = Slower convergence, vice versa
OPTIMIZER = Adam(learning_rate=LEARNING_RATE) # Much better at escaping minina and more stable based on research and experimentation
BATCH_SIZE = 10
EPOCHS = 300

def file_to_nparray(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        data = []

        for line in lines:
            value = float(line.strip())
            data.append(value)
    return np.array(data)

def chart_data_summary(biomass,substrate,compoundA):
    x = np.arange(0,len(biomass))
    plt.figure(figsize=(30,5))
    plt.plot(x, biomass, label='Biomass')
    plt.plot(x, substrate, label='Substrate')
    plt.plot(x, compoundA, label='CompoundA')
    plt.xlabel('t')
    plt.ylabel('grams')
    plt.title('Base Dataset')
    plt.legend()
    plt.show()

def pre_process(array):
    # Clean up data
    def remove_negatives(array):
        clean_array = np.copy(array)
        for i in range(0, len(array)):
            if array[i] < 0:
                clean_array[i] = 0 #replace negatives with 0
                # Another strategy is to offset the difference onto Biomass since the mass of the system should remain constant , e.g. system_mass = substrate + biomass
        return clean_array

    def clean_array(array):    
        array = remove_negatives(array)
        #optional pre-processing
        #array = min_max_normalization(array) #normalisation optional since same units used for all data
        #Regularization - prevents overfitting and improves generalization of ML model - adds a penalty to the loss function. L1 - reduces weights of non-important features, L2 - adds penalty when too many weights
        #Scaling
        #Outliers
        return array
    
    # Split Training data
    def split_array(array, split = TRAINING_SPLIT): 
        train_size = int(len(array) * split)
        train_array = array[:train_size]
        test_array = array[train_size:]
        return train_array, test_array
    
    train_set , test_set = split_array(clean_array(array))
    return train_set,test_set

# Combine compoundA and substrate as the two inputs into our Neural Network
def combine_inputs(a, b):
    c = []
    for i in range(len(a)):
        c.append([a[i], b[i]])
    return np.array(c)

## Metrics
def chart_output_vs_target(predicted, observed, chart_title='Observed vs Predicted Biomass'):
    x = np.arange(0,len(predicted))
    plt.figure(figsize=(30,5))

    plt.plot(x, predicted, label='Estimated Biomass') 
    plt.plot(x, observed, label='Real Biomass')
    
    plt.title(chart_title)
    plt.legend()
    plt.show() #######################

def chart_loss_accuracy(predicted, observed, chart_title='Observed vs Predicted Loss / Accuracy'):
    loss = []
    for i in range(predicted.size):
        squared_error = (observed[i] - predicted[i]) ** 2
        loss.append(squared_error)

    x = np.arange(0,len(predicted))
    plt.figure(figsize=(30,5))

    plt.plot(x, observed, label='Actual Biomass')
    plt.plot(x, predicted, label='Estimated Biomass') 
    plt.plot(x, loss, label='loss (squared error)', color='Red')
    
    plt.title(chart_title)
    plt.legend()

def calc_ia(observed, predicted):
    n = len(observed)
    num = 0
    den = 0
    om = np.mean(observed)
    for i in range(0, n):
        num += (observed[i] - predicted[i]) ** 2
        den += (abs(observed[i]-om) + abs(predicted[i]-om)) ** 2
    ia = 1 - (num/den)
    return float(ia)

def calc_rms(observed, predicted):
    n = len(observed)
    num = 0
    den = 0
    for i in range(0, n):
        num += (observed[i] - predicted[i]) ** 2
        den += observed[i]**2
    rms = math.sqrt(num/den)
    return float(rms)

def calc_rsd(observed, predicted):
    n = len(observed)
    num = 0
    den = n
    for i in range(0, n):
        num += (observed[i] - predicted[i]) ** 2
    rsd = math.sqrt(num/den)
    return float(rsd)

def calculate_performance(predicted, observed, test_name):
    performance = {
        "test_name": test_name,
        "performance_metrics":{
        "IA": calc_ia(observed, predicted),
        "RMS": calc_rms(observed, predicted),
        "RSD": calc_rsd(observed, predicted)
        }
    }
    print(f'-- Model Performance --')
    print(f'{performance["test_name"]}')
    print(f'IA = {performance["performance_metrics"]["IA"]}')
    print(f'RMS = {performance["performance_metrics"]["RMS"]}')
    print(f'RSD = {performance["performance_metrics"]["RSD"]}')
    return performance

def create_model(): #This is the baseline layer...
    nnet = Sequential() 
    # 2 Units in the input layer for Compound A and substrate + 10 in hidden layer
    nnet.add(Dense(10, input_dim=2, activation='linear')) # 2 input dimensions
    # Hidden Layer - #best practices here is 
    nnet.add(Dense(8, activation='sigmoid'))
    # One output layer
    nnet.add(Dense(1, activation='linear')) # Commonly used in regression problems - preferred because output should be continuous whereas reLu introduces non-linearity resulting in poor output
    print(nnet.summary())
    return nnet

# Train Model: 
def compile_model(model):
    #Compile model
    model.compile(
    optimizer=OPTIMIZER, 
    loss=LOSS_FUNCTION,
    metrics=['accuracy']
    )
    return model

def metric_summary(mode,stage,model,observed_data,input_data,model_performance):
    observed = observed_data
    predicted = model.predict(input_data)
    performance = calculate_performance(predicted, observed, f'{mode} set {stage}-Training')
    # model_performance.append(performance)
    # chart_output_vs_target(predicted, observed, f'{mode} set - Real Biomass vs Estimated Biomass ({stage} Training)')

# Show relationship:
def plot_loss_accuracy(history):
    # Charts overlaid to show the relationship between Loss and Accuracy over time
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel("Epoch")
    ax1.set_title("Model Loss (MSE) / Accuracy")
    lns1 = ax1.plot(history.history['loss'], label='loss', color='Red')
    ax1.set_ylabel("Loss")
    lns2 = ax2.plot(history.history['accuracy'], label='Accuracy', color='Green')
    ax2.set_ylabel("Accuracy")
    lns = lns1 + lns2
    ax1.legend(lns, ['loss', 'accuracy'])
    plt.show() #######################

# Compute Error Index:
def chart_compute_error(pre_training, post_training, title='Model Error'):
    # Training Set Performance
    print("Pre Training")
    print(pre_training)
    print("Post Training")
    print(post_training)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    axes[0].bar('Before', pre_training['IA'])
    axes[0].bar('After', post_training['IA'])
    axes[0].set_title('IA (Target: IA > 0.9)')
    axes[0].annotate(str(pre_training['IA']), (-0.3, pre_training['IA'] + 0.01), color='black')
    axes[0].annotate(str(post_training['IA']), (0.65, post_training['IA'] + 0.01), color='black')

    axes[1].bar('Before', pre_training['RMS'])
    axes[1].bar('After', post_training['RMS'])
    axes[1].set_title('RMS (Target: RMS < 0.1)')
    axes[1].annotate(str(pre_training['RMS']), (-0.3, pre_training['RMS'] + 0.01), color='black')
    axes[1].annotate(str(post_training['RMS']), (0.65, post_training['RMS'] + 0.01), color='black')

    axes[2].bar('Before', pre_training['RSD'])
    axes[2].bar('After', post_training['RSD'])
    axes[2].set_title('RSD (Target: RSD < 0.3)')
    axes[2].annotate(str(pre_training['RSD']), (-0.3, pre_training['RSD'] + 0.01), color='black')
    axes[2].annotate(str(post_training['RSD']), (0.65, post_training['RSD'] + 0.01), color='black')
    fig.canvas.manager.set_window_title(title)
    plt.show()

def main():
    #Load data
    biomass = file_to_nparray("biomass.txt")
    compoundA = file_to_nparray("compoundA.txt")
    substrate = file_to_nparray("substrate.txt")

    #Check Data
    pre_process_data = pd.DataFrame(np.column_stack((biomass,compoundA,substrate)),columns=["biomass","compoundA","substrate"]).describe()
    print(pre_process_data)
    print('\n')

    #Plot Data -- Issues with this
    chart_data_summary(biomass,substrate,compoundA)

    #Pre-process Data + Clean
    biomass_train, biomass_test = pre_process(biomass)
    compoundA_train, compoundA_test = pre_process(compoundA)
    substrate_train, substrate_test = pre_process(substrate)

    #Combine input
    inputs_train = combine_inputs(compoundA_train, substrate_train)
    inputs_test = combine_inputs(compoundA_test, substrate_test)

    #Create Model and compile
    nn_model = create_model()
    nn_model = compile_model(nn_model) 

    #Set up Model Capture 
    model_performance = []

    #Show Pre-training metrics
    metric_summary('Train','Pre',nn_model,biomass_train,inputs_train,model_performance)
    metric_summary('Test','Pre',nn_model,biomass_test,inputs_test,model_performance)

    # Charting Training Set Without Training
    observed = biomass_train
    predicted = nn_model.predict(inputs_train)
    performance = calculate_performance(predicted, observed, 'Training set Pre-Training')
    model_performance.append(performance)
    # chart_output_vs_target(predicted, observed, 'TRAIN - Real Biomass vs Estimated Biomass (Without Training)')
    chart_loss_accuracy(predicted, observed, 'Loss before training [TRAINING SET]')

    # Charting Test Set Without Training
    observed = biomass_test
    predicted = nn_model.predict(inputs_test)
    performance = calculate_performance(predicted, observed, 'Test set Pre-Training')
    model_performance.append(performance)
    # chart_output_vs_target(predicted, observed, 'TEST - Real Biomass vs Estimated Biomass (Without Training)')
    chart_loss_accuracy(predicted, observed, 'Loss before training [TEST SET]')

    #Train model:
    history = nn_model.fit(inputs_train, biomass_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    #Show Post-training metrics
    metric_summary('Train','Post',nn_model,biomass_train,inputs_train,model_performance)
    metric_summary('Test','Post',nn_model,biomass_test,inputs_test,model_performance)

    # Checking how we perform on the training set --> check potential overfitting
    observed = biomass_train
    predicted = nn_model.predict(inputs_train)
    performance = calculate_performance(predicted, observed, 'Training set After Training')
    model_performance.append(performance)
    # chart_output_vs_target(predicted, observed, 'Real Biomass vs Estimated Biomass (After Training)')
    chart_loss_accuracy(predicted, observed, 'Loss after training [TRAINING SET]')

    # Charting Test Set After Training
    observed = biomass_test
    predicted = nn_model.predict(inputs_test)
    performance = calculate_performance(predicted, observed, 'Test set After Training')
    model_performance.append(performance)
    # chart_output_vs_target(predicted, observed, 'Real Biomass vs Estimated Biomass (After Training)')
    chart_loss_accuracy(predicted, observed, 'Loss after training [TEST SET]')

    #Plot loss accuracy relationship
    plot_loss_accuracy(history)

    #Compute error index
    print(model_performance)
    chart_compute_error(model_performance[0]['performance_metrics'],model_performance[2]['performance_metrics'], 'Training Set')
    chart_compute_error(model_performance[1]['performance_metrics'],model_performance[3]['performance_metrics'], 'Test Set')

if __name__ == "__main__":
    main()