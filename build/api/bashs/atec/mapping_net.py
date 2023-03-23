'特征适应变换'
import os
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model  
from contextlib import redirect_stdout

# 特征映射
def fea_mapping(path, train_x, train_y, test_x, test_y, epoch, batch_size):
    fitting_model = keras.Sequential([
        keras.layers.Conv1D(30, kernel_size=5, padding='valid', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPool1D(pool_size=2, strides=2),

        keras.layers.Conv1D(25, kernel_size=5, padding='valid', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPool1D(pool_size=2, strides=2),

        keras.layers.Conv1D(15, kernel_size=5, padding='valid', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPool1D(pool_size=2, strides=2),

        keras.layers.Flatten(),

        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation='softmax'),
        keras.layers.Dense(len(train_y[0]))
    ])
    fitting_model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])
    learn_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='lr', factor=0.99, patience=3,
                                                             verbose=0, min_lr=0.00001)
    checkpoint = keras.callbacks.ModelCheckpoint(path, monitor='val_loss', verbose=0,
                                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint, learn_rate_reduction]
    fitting_model.fit(train_x, train_y, batch_size=batch_size, epochs=epoch, validation_data=(test_x, test_y),
                      callbacks=callbacks_list, verbose=0, validation_freq=1)
    test_model = keras.models.load_model(path)
    saveModelInfo(test_model, path)
    train_pred = test_model.predict(train_x)
    test_pred = test_model.predict(test_x)

    return train_pred, test_pred


# 保存模型结构信息
def saveModelInfo(model, modelPath):
    rootPath = os.path.dirname(modelPath)
    modelName = os.path.basename(modelPath).split('.')[0]

    # 保存模型所有基本信息
    with open(rootPath + '/'+ modelName + "_modelInfo.txt", 'w') as f:
        with redirect_stdout(f):
            model.summary(line_length=200, positions=[0.30,0.60,0.7,1.0])
        
    # 保存模型所有层的名称至xml文件
    from xml.dom.minidom import Document
    xmlDoc = Document()
    child_1 = xmlDoc.createElement(modelName)
    xmlDoc.appendChild(child_1)
    child_2 = xmlDoc.createElement(modelName)
    child_1.appendChild(child_2)
    for layer in model.layers:  
        layer = layer.name.replace("/", "_")
        nodeList = layer.split("_")
        for i in range(len(nodeList)):
            modeName = nodeList[i].strip()
            if modeName.isdigit():
                modeName = "_" + modeName
            if i == 0:
                # 如果以modeName为名的节点已经存在，就不再创建，直接挂
                if len(child_2.getElementsByTagName(modeName)) == 0:
                    node1 = xmlDoc.createElement(modeName)
                    child_2.appendChild(node1)
                else:
                    node1 = child_2.getElementsByTagName(modeName)[0]
            elif i == 1:
                if len(node1.getElementsByTagName(modeName)) == 0:
                    node2 = xmlDoc.createElement(modeName)
                    node1.appendChild(node2)
                else:
                    node2 = node1.getElementsByTagName(modeName)[0]
            elif i == 2:
                if len(node2.getElementsByTagName(modeName)) == 0:
                    node3 = xmlDoc.createElement(modeName)
                    node2.appendChild(node3)
                else:
                    node3 = node2.getElementsByTagName(modeName)[0]
            elif i == 3:
                if len(node3.getElementsByTagName(modeName)) == 0:
                    node4 = xmlDoc.createElement(modeName)
                    node3.appendChild(node4)
                else:
                    node4 = node3.getElementsByTagName(modeName)[0]
    f = open(rootPath + '/'+ modelName + "_struct.xml", "w")
    xmlDoc.writexml(f, addindent='\t', newl='\n', encoding="utf-8")
    f.close()

    # 保存模型结构图
    if not os.path.exists(rootPath + '/'+ modelName + "_structImage"):
        os.makedirs(rootPath + '/'+ modelName + "_structImage")
    plot_model(model, to_file = rootPath + '/'+ modelName + "_structImage/framework.png", show_shapes=True, show_layer_names=True, dpi=800)
