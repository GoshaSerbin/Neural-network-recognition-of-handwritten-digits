import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.datasets import mnist
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

#функция активация для внутренних нейронов, гиперболический тангенс со значениями (0,1)
def sigma_tg(x): 
    return 1/(1 + np.exp(-x))

#ее производная
def dsigma_tg(x):
    return x*(1-x)

def sigma_log(x,vec):
    return np.exp(x)/np.sum(np.exp(vec))

#количество входных нейронов, 28x28 пикселей
size_of_input = 28 * 28

#количество выходных нейронов, 10 цифр
size_of_output = 10

#количество внутренних слоев
num_of_hidden_layers = 2

#количество нейронов на внутренних слоях (для всех одинаковое)
num_of_neurons_in_hidden_layer = 70

#смещения на выходных нейронах
final_biases = np.random.random(size_of_output)

#смещения на внутренних нейронах
biases_in_hidden_layers = np.random.random( (num_of_hidden_layers,num_of_neurons_in_hidden_layer))

#выбираеем функции активации для скрытых слоев и выходного слоя
act_hidden = sigma_tg
dact_hidden = dsigma_tg

act_final = sigma_log
dact_final = dsigma_tg

#веса на выходных слоях, в диапозоне (-1,1)
weights_in_hidden_layers = 2 * np.random.random((num_of_hidden_layers-1, num_of_neurons_in_hidden_layer,num_of_neurons_in_hidden_layer)) - 1

#веса на входном слое
init_weights = 2 * np.random.random((num_of_neurons_in_hidden_layer,size_of_input)) - 1

#веса на выходном слое
final_weights = 2 * np.random.random((size_of_output,num_of_neurons_in_hidden_layer)) - 1

#пропускает вектор наблюдений через нейронную сеть
def go_forward(inp):    

    #сохраняем результат прогонки для каждого слоя в массив out для дальнейшей коррекции весов
    out  = np.zeros((num_of_hidden_layers,num_of_neurons_in_hidden_layer))
    sum = np.dot(init_weights,inp) + biases_in_hidden_layers[0]
    out[0] = np.array([act_hidden(x) for x in sum])

    for i in range(1, num_of_hidden_layers):
        sum = np.dot(weights_in_hidden_layers[i-1],out[i-1]) + biases_in_hidden_layers[i]
        out[i] = np.array([act_hidden(x) for x in sum])

    sum = np.dot(final_weights,out[num_of_hidden_layers - 1]) + final_biases
    y = np.array([act_final(x,sum) for x in sum])

    return (y,out)

def train(epoch_x,epoch_y):

    global init_weights, weights_in_hidden_layers, final_weights,biases_in_hidden_layers,final_biases,iteration
    
    count = len(epoch_y) #размер всей эпохи
    mini_batch = 1 #размер minibatch
    E = np.identity(size_of_output) #для преобразования числа в вектор

    #оптимизация с помощью метода инерции
    final_h=np.zeros(final_weights.shape)
    final_bh=np.zeros(final_biases.shape)
    h_in_hidden_layers=np.zeros(weights_in_hidden_layers.shape)
    bh_in_hidden_layers=np.zeros(biases_in_hidden_layers.shape)
    init_h=np.zeros(init_weights.shape)
    alpha = 0.1

    for k in range(count//mini_batch):
        iteration += 1
        lmd = 0.1 / pow(iteration, 0.1)
        final_delta = np.zeros(size_of_output) #вектор локального градиента на выходном слое
        delta_in_hidden_layers = np.zeros((num_of_hidden_layers,num_of_neurons_in_hidden_layer)) #на промежуточных слоях
        for m in range(mini_batch): # m раз только вычисляем градиенты delta и не меняем веса
            rn = np.random.randint(0 , count) #np.random.randint(k*mini_batch , (k+1)*mini_batch) # k*mini_batch + m
            x = epoch_x[rn]
            y_true = E[epoch_y[rn]]
            y, out = go_forward(x)

            delta = y-y_true
            final_delta += delta #vector
            delta = np.dot(delta,final_weights) * dact_hidden(out[-1])
            delta_in_hidden_layers[-1] += delta
            for i in range(1,num_of_hidden_layers):   
                delta = np.dot(delta,weights_in_hidden_layers[-i]) * dact_hidden(out[-i-1])
                delta_in_hidden_layers[-i-1] += delta

        #меняем веса  
        for i in range(size_of_output):   
            final_h[i,:] = alpha*final_h[i,:] + lmd * final_delta[i] * out[-1]
            final_bh[i] = alpha*final_bh[i] + lmd * final_delta[i]           
        final_weights -= final_h
        final_biases -= final_bh

        for i in range(1,num_of_hidden_layers):  
            for j in range(num_of_neurons_in_hidden_layer): 
                h_in_hidden_layers[-i,j,:] = alpha * h_in_hidden_layers[-i,j,:] + lmd * delta_in_hidden_layers[-i,j] * out[-i-1]
                bh_in_hidden_layers[-i,j] = alpha * bh_in_hidden_layers[-i,j] + lmd * delta_in_hidden_layers[-i,j]
                weights_in_hidden_layers[-i,j,:] -= h_in_hidden_layers[-i,j,:]
                biases_in_hidden_layers[-i,j] -= bh_in_hidden_layers[-i,j]

        for i in range(num_of_neurons_in_hidden_layer):        
            init_h[i,:] = alpha*init_h[i,:] + lmd * delta_in_hidden_layers[0,i] * np.array(x)  
            bh_in_hidden_layers[0,i] =  alpha *bh_in_hidden_layers[0,i] + lmd * delta_in_hidden_layers[0,i]
            init_weights[i,:] -= init_h[i,:] 
            biases_in_hidden_layers[0,i] -= bh_in_hidden_layers[0,i]
        

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

epoch_X = np.zeros((len(train_X),28*28))
epoch_test_X = np.zeros((len(test_X),28*28))

# преобразуем входные данные
for i in range(len(train_X)):
    epoch_X[i] = train_X[i].flatten()/255
    
for i in range(len(test_X)):    
    epoch_test_X[i] = test_X[i].flatten()/255


#тренируем
start_time = time.perf_counter()
accuracy = []
errors = []
coord_x = []
iteration = 0
for k in range(15):   
    train(epoch_X,train_y)
    total_time  = time.perf_counter() - start_time
    print(f"Era {k} is done!\nTotal time: {total_time:0.2f}. Calculating accuracy: ...")
    counter = 0
    # проверка
    error = 0
    for i in range(len(epoch_test_X)):
        y,out  = go_forward(epoch_test_X[i])
        error -= np.sum(np.log(y[test_y[i]]))
        ind = np.argmax(y)
        if test_y[i] == ind : counter +=1 
    error /= len(epoch_test_X)
    error *= 20
    accuracy.append(counter/len(epoch_test_X) * 100)
    errors.append(error)
    coord_x.append(k+1)
    print("Accuracy: ", accuracy[-1], "%\n")

fig = plt.figure(figsize = (7,4))
ax = fig.add_subplot()
ax.set_title('Handwritten digit recognition',fontsize = 16,fontname="Times New Roman")
ax.set_xlabel('Era',fontname="Times New Roman")
ax.set_ylabel('%',fontname="Times New Roman")
ax.plot(coord_x,np.array(accuracy),'-bo',label = 'Accuracy') 
ax.plot(coord_x,np.array(errors),'--rv',label = 'Error') 
ax.xaxis.set_major_locator(MultipleLocator(base = 1))
ax.legend(loc = 'right')
plt.grid()
plt.show()

np.save('init_weights',init_weights)
np.save('weights_in_hidden_layers',weights_in_hidden_layers)
np.save('final_weights',final_weights)    
np.save('biases_in_hidden_layers',biases_in_hidden_layers)
np.save('final_biases',final_biases)






