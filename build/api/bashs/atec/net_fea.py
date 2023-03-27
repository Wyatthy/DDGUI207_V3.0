'''神经网络特征提取'''
import os
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model  
from contextlib import redirect_stdout

def net_fea_extract(path, train_x, train_y, test_x, test_y, epoch, batch_size):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(16, 8, strides=2, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2))

    model.add(keras.layers.Conv1D(32, 4, strides=2, activation='relu', padding="valid"))
    model.add(keras.layers.MaxPooling1D(2))

    # model.add(keras.layers.Conv1D(64, 4, strides=2, activation='relu', padding="valid"))
    # model.add(keras.layers.MaxPooling1D(2))

    model.add(keras.layers.Conv1D(32, 2, strides=1, activation='relu', padding="valid"))
    model.add(keras.layers.MaxPooling1D(2))

    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(len(train_y[0]), activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    learn_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='lr', factor=0.99, patience=3,
                                                             verbose=0, min_lr=0.0001)
    checkpoint = keras.callbacks.ModelCheckpoint(path, monitor='val_accuracy', verbose=0,
                                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint, learn_rate_reduction]
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epoch, shuffle=True,
              validation_data=(test_x, test_y), callbacks=callbacks_list, verbose=0, validation_freq=1)
    test_model = keras.models.load_model(path)
    saveModelInfo(test_model, path)
    functor = keras.models.Model(inputs=test_model.input, outputs=test_model.layers[-2].output)  # 输出模型倒数第二层
    train_fea = functor.predict(train_x)
    test_fea = functor.predict(test_x)

    return train_fea, test_fea



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
    f = open(rootPath + '/'+ modelName + "_struct.xml", "w", encoding='utf-8')
    xmlDoc.writexml(f, addindent='\t', newl='\n', encoding="utf-8")
    f.close()

    # 保存模型结构图
    if not os.path.exists(rootPath + '/'+ modelName + "_structImage"):
        os.makedirs(rootPath + '/'+ modelName + "_structImage")
    plot_model(model, to_file = rootPath + '/'+ modelName + "_structImage/framework.png", show_shapes=True, show_layer_names=True, dpi=800)
