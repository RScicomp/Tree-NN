import numpy
import tensorflow as tf
from tensorflow import keras
import random
import pandas as pd
import numpy as np

"""### Generate Mock data"""

distance_definitions = {'near':.95,'medium':.7} # averaged output by perceptron
wheel_definitions = {'slow':.5,'medium':.8}
near_sensor = lambda x: 0 if (x >= distance_definitions['near']) else ( 1 if x >= distance_definitions['medium'] else 2)
measure_wheel = lambda x: 0 if (x < wheel_definitions['slow']) else ( 1 if x < wheel_definitions['medium'] else 0)
turn_away = lambda x: 0 if (near_sensor(x)==0) else 2
actions = {'left':0,'right':1,'forward':2}

#near, medium, far - 0,1,2
# To implement next - when torque is greater, make sensor data change
generated_points = 1000

def generate_sensor_data(evaluation):
    # randomly generate voltage value
    # perceptron tends to over-estimate how close something is to compensate
    # for any signal loss due to noise/angle
    # skew probability to produce smaller numbers more often, to balance out this trend
    sensor_data_front = [random.triangular(0,2.9,0) for x in range(generated_points)]
    sensor_data_front_target = [evaluation(x) for x in sensor_data_front]
    sensor_data_front = pd.DataFrame(zip(sensor_data_front,sensor_data_front_target))
    sensor_data_front.columns = ['x','y']
    return(sensor_data_front)


def generate_decisions(evaluation,turnaway):
    sensor_data_front = [random.triangular(0,2.9,0) for x in range(generated_points)]
    sensor_data_front_target = [evaluation(x) for x in sensor_data_front]
    sensor_data_decision = [turnaway(x) for x in sensor_data_front]
    sensor_data_front = pd.DataFrame(zip(sensor_data_front,sensor_data_decision,sensor_data_front_target))
    sensor_data_front.columns = ['x','decision','y']
    return(sensor_data_front)

sensor_front_data = generate_sensor_data(evaluation=near_sensor)
sensor_front_data_decision = generate_decisions(evaluation=near_sensor,turnaway=turn_away)

sensor_front_data_decision

# wheel_lf_data = generate_sensor_data(evaluation=measure_wheel)
# wheel_rf_data = generate_sensor_data(evaluation=measure_wheel)
# wheel_lb_data = generate_sensor_data(evaluation=measure_wheel)
# wheel_rb_data = generate_sensor_data(evaluation=measure_wheel)
# from keras.utils.np_utils import to_categorical

class NeuralNet():
    def __init__(self,x,y,output,name):
        self.X = x
        self.y = y
        self.output=output
        self.name=name
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(20,activation='relu')) # A layer connected to all layers
        model.add(keras.layers.Dense(20,activation='relu')) # A layer connected to all layers
        model.add(keras.layers.Dense(output,activation='softmax'))

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),#'sparse_categorical_cross_entropy',
            optimizer=keras.optimizers.SGD(),
            metrics=['accuracy']
        )

        self.model=model
    def train(self,**args):
        pass


    def predict(self,test,**args):
        predictions=self.model.predict(test)
        return(predictions)

    def cat_predict(self,test,**args):
        predictions=self.model.predict_classes(test)
        return(predictions)


class InfraredNet(NeuralNet):
    def __init__(self,input,target,output,name):
        super().__init__(input,target,output,name)
        model = keras.models.Sequential()
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(20,activation='relu')) # A layer connected to all layers
        model.add(keras.layers.Dense(20,activation='relu')) # A layer connected to all layers
        model.add(keras.layers.Dense(output,activation='softmax'))

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),#'sparse_categorical_cross_entropy',
            optimizer=keras.optimizers.SGD(),
            metrics=['accuracy']
        )
        # input_sensor = keras.layers.Input(shape=input.shape[1], name=name)
        # hidden1 = keras.layers.Dense(30, activation="relu")(input_sensor)
        # output = keras.layers.Dense(len(set(self.y)), name="output")(hidden1)

        # self.model = keras.models.Model(inputs=[input_sensor], outputs=[output])
        # self.model.compile(loss="sparse_categorical_entropy",
        #               optimizer=keras.optimizers.RMSprop(),
        #               metrics=['accuracy'])
        self.model=model


    def train(self):
        self.model.fit(self.X,self.y,epochs=30,batch_size=32)

    # def predict(self,test):
    #   self.model.predict(test)


class MainNet(NeuralNet):
    def __init__(self,input,target,output,name):
        super().__init__(input,target,output,name)
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(20,activation='relu')) # A layer connected to all layers
        model.add(keras.layers.Dense(20,activation='relu')) # A layer connected to all layers
        model.add(keras.layers.Dense(output,activation='softmax'))

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),#'sparse_categorical_cross_entropy',
            optimizer=keras.optimizers.SGD(),
            metrics=['accuracy']
        )

        self.model=model


    def train(self):
        print(self.X.shape,self.y.shape)
        self.model.fit(self.X,self.y,epochs=30,batch_size=32)

    # def predict(self,test):
    #   self.model.predict(self,test)

class TreeNet():
    def __init__(self,sensors,mainnet=None):
        # assert(type(models[0]) == NeuralNet)
        self.sensors = sensors

        self.mainnet = mainnet


    # Train all models
    def train(self,data,y,train_sensors=True):
        all_predictions = []

        for model,sensor_data in zip(self.sensors,data):
            if(train_sensors==True):
                print("Training " + model.name)
                model.X=sensor_data[0]
                model.y=sensor_data[1]
                model.train()
            predictions = model.predict(sensor_data[0])
            all_predictions.append(predictions)
        # all_predictions
        training_data=np.concatenate((all_predictions),axis=1)
        print(training_data.shape)
        self.mainnet.X = training_data
        self.mainnet.y = y
        print("Training mainnet...")
        self.mainnet.train()
        # return(training_data)#training_data)


    def standardize_data(self,X):
        return([[x] for x in X])


    def standardize_data_all(self,X):
        all_data=[]
        for x in X:
            res=[]
            for sub_x in x:
                res.append([sub_x])
            all_data.append(res)
        return(all_data)

    # Use predictions of models to train main net
    def predict(self,X):
        all_predictions = []

        for model,sensor_data in zip(self.sensors,X):
            predictions = model.predict(sensor_data)
            all_predictions.append(predictions)

        # all_predictions
        test_data=np.concatenate((all_predictions),axis=1)
        # print(training_data)
        predictions=self.mainnet.predict(test_data)
        return(predictions)

    def predict_class(self,X):
        all_predictions = []

        for model,sensor_data in zip(self.sensors,X):

            predictions = model.predict(sensor_data)
            all_predictions.append(predictions)
        # all_predictions
        test_data=np.concatenate((all_predictions),axis=1)
        # print(test_data)
        # print(training_data)
        predictions=self.mainnet.cat_predict(test_data)
        return(predictions)


# defining training set for infrared nets --------------------------------------------------------------------
train_x = sensor_front_data_decision[['x']].values
train_y = sensor_front_data_decision[['y']].values
# train = []
# for i in range(7):
#     train.append((train_x, train_y))


# defining each sensor ---------------------------------------------------------------------------------------
infr_net_l = InfraredNet(input=train_x,target = train_y,output=3,name='left_sensor')
infr_net_lt = InfraredNet(input=train_x,target = train_y,output=3,name='left_tilt_sensor')

infr_net_fl = InfraredNet(input=train_x,target = train_y,output=3,name='front_left_sensor')
infr_net_fc = InfraredNet(input=train_x,target = train_y,output=3,name='front_center_sensor')
infr_net_fr = InfraredNet(input=train_x,target = train_y,output=3,name='front_right_sensor')

infr_net_rt = InfraredNet(input=train_x,target = train_y,output=3,name='right_tilt_sensor')
infr_net_r = InfraredNet(input=train_x,target = train_y,output=3,name='right_sensor')


infrared_net_list = [infr_net_l,infr_net_lt,infr_net_fl,infr_net_fc,infr_net_fr,infr_net_rt,infr_net_r]
main_net = MainNet(input=None,target=None,output=3,name="main_net")


for net in infrared_net_list:
    net.train()
# -------------------------------------------------------------------------------------------------------------
# set up tree net, train tree net with manually collected data
tree_net = TreeNet(infrared_net_list,mainnet=main_net)

# data reformatting -------------------------------------------------------------------------------------------
# problem: "[0,0,0,0,0,0,0]" keeps being interpreted as string instead of array
# either try to convert to right type or change store type
from ast import literal_eval
from data_process import get_float_array

df = pd.read_csv("dataset.csv")

x0 = df[['s0']].values
x1 = df[['s1']].values
x2 = df[['s2']].values
x3 = df[['s3']].values
x4 = df[['s4']].values
x5 = df[['s5']].values
x6 = df[['s6']].values

y0 = [near_sensor(i) for i in x0]
y1 = [near_sensor(i) for i in x1]
y2 = [near_sensor(i) for i in x2]
y3 = [near_sensor(i) for i in x3]
y4 = [near_sensor(i) for i in x4]
y5 = [near_sensor(i) for i in x5]
y6 = [near_sensor(i) for i in x6]

df['action'] = [int(i) for i in df['action']]
# df['action'] = [2 if x == -1 else x for x in df['action']]
# df.to_csv("dataset.csv")
truth = df[['action']].values


# truth = df.action.apply(float)
train = [(x0,y0), (x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6)]
print("train shape")
print(len(train))
print("truth shape")
print(truth.shape)
print(len(df['action']))

tree_net.train(data=train, y=truth, train_sensors=False)
print('predicting:')
print(tree_net.predict([[2.2],[0],[0],[0],[0],[0],[0]]))

# from sklearn.metrics import accuracy_score
predictions=tree_net.predict_class(test)

# print(accuracy_score(sensor_front_data_decision_test['decision'],predictions))
# tree_net.standardize_data_all([[1.028258,1.028258],[.31,.31]])


class GameWrapper():
    def __init__(self,treenet):
        self.model = treenet
        self.orientation = 0
        self.obstacle = 0
        self.position = [0,0]

    def generateObstacle(self):
        pass

    def generateState(self):
        data= []
        for model in self.model.sensors:
            if("infrared" in model.name):
                sensor = random.random()
                #crash= if random.random() > .5
                #data.append([random.random(),crash])

            # else:
            #   model.predict(random.random())
        self.state= data
    def act(self):
        pass
    def predict(self):
        prediction = self.model.predict(self.state)
        return(prediction)


# Wheel data ()
# Torque, orientation, speed.
# IF all four wheels aren't spinning etc...
# It's like having a GPS and GPS position doesn't change.

#touch: A measurement of resistance (continious - whiskers bend [0,1])
#Not touching, touching lightly, Touching a lot
# touch_data_back = [[0.1,0],[0.1,1]]
# touch_data_back = pd.DataFrame(touch_data_back)
# touch_data_back.columns = ['touch','time']
#
# touch_data_front = [[0.1,0],[0.1,1]]
# touch_data_front = pd.DataFrame(touch_data_front)
# touch_data_front.columns = ['touch','time']

"""### Multiple input Model"""

#Train leaf all by itself, lock the leaves. Still send data through them, their output is used to train the next layer. Lock those layers.
#Combining touch and whether wheels were turning: Can conclude that the object im pushing on is not moveable. Abstract information gathering*
#Come up with a data table and send it into the robot to produce actions.


# input_front_sensor = keras.layers.Input(shape=sensor_data_front.shape[1], name="front_sensor_input")
# input_front_touch = keras.layers.Input(shape=touch_data_front.shape[1], name="front_touch_sensor_input")
# input_back_sensor = keras.layers.Input(shape=sensor_data_front.shape[1], name="back_sensor_input")
# input_back_touch = keras.layers.Input(shape=sensor_data_front.shape[1], name="back_touch_sensor_input")
#
# hidden1 = keras.layers.Dense(30, activation="relu")(input_front_sensor)
# hidden2 = keras.layers.Dense(1, activation="relu")(hidden1)
# hidden3 = keras.layers.Dense(30, activation="relu")(input_back_sensor)
# hidden4 = keras.layers.Dense(1, activation="relu")(hidden3)
# concat = keras.layers.concatenate([hidden2,hidden4])
# hidden5 = keras.layers.Dense(30,activation="relu")(concat)
#
# output = keras.layers.Dense(5, name="output")(hidden5)
#
#
# model = keras.models.Model(inputs=[input_front_sensor,input_back_sensor], outputs=[output])
#
# keras.utils.plot_model(model, "test.png", show_shapes=True)