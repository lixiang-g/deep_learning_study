import tensorflow as tf
import numpy as np

# --- 1. 准备数据 ---
print("准备数据...")
# 输入数据 X: 包含四个输入对
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
# 目标输出 y: 对应异或操作的结果
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# --- 2. 定义神经网络模型 ---
print("定义模型结构...")
# 我们使用Functional API，因为它更明确和灵活
inputs = tf.keras.Input(shape=(2,), name='Input_Layer')
# 隐藏层：4个神经元，使用ReLU激活函数引入非线性
hidden_out = tf.keras.layers.Dense(units=4, activation='relu', name='Hidden_Layer')(inputs)
# 输出层：1个神经元，使用Sigmoid激活函数将输出转换为0-1之间的概率
outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name='Output_Layer')(hidden_out)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# 打印模型摘要
model.summary()

# --- 3. 编译模型 ---
print("\n编译模型...")
# 我们需要定义优化器、损失函数和评估指标
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- 4. 训练模型 ---
print("开始训练模型...")
# `fit`函数会自动完成前向传播、计算损失、反向传播和更新权重的循环
model.fit(X, y, epochs=1000, verbose=0) # verbose=0表示训练时不打印过程
print("模型训练完成！")

# --- 5. 评估与测试 ---
print("\n评估模型效果...")
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"最终准确率: {accuracy * 100:.2f}%")

print("\n模型对四个输入的预测结果:")
predictions = model.predict(X)
for i in range(len(X)):
    print(f"输入: {X[i]}, 真实值: {int(y[i][0])}, 预测概率: {predictions[i][0]:.4f}, 预测结果: {int(predictions[i].round())}")
