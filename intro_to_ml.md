# Introduction to Machine Learning

### AI, MACHINELEARNING AND DEEPLEARNING

人工智慧是我们想要达成的目标，而机器学习是想要达成目标的手段，深度学习就是机器学习的其中一种方法。

### Steps of ML
我们想要让机器变聪明，那么机器就要学会自己去学习。以supervised learning为例，也就是说我们给机器一些data，机器通过对data的学习进行举一反三，达到学会更多东西的效果。把机器当作一个小孩子一样去教它，我们不是只是单纯的让他记忆知识，而是让它知道怎样的输入对应怎样的输出是正确的，然后去寻找输入和输出之间的关系，那下次他遇到类似的输入时能做出正确的输出。就比如说给他看一只猫的图像，告诉他这是一只猫，之后再给他看另一张猫的图像（有可能是不同品种的猫），他也能回答出来【哦，这是一只猫】。

我们假定机器他拥有好多个function，那么整个function set我们把它叫做model。通过我们输入的training set，机器来判断整个model里，哪个function是好的哪个function有偏差，即判断goodness of function，通过高效的演算法，最终选一个最好的$f^{*}$。那这一段就是training。为了验证我们选的$f^{*}$在能否在实际场景中有效运作，我们就要进行testing这一过程。比如我们要把这个训练好function应用到影像识别中，我们给机器一张它没见过的图像并且不告诉他答案，如果他能够判断正确，那说明这个function是有一定正确率的。整个机器学习的过程大致如下三个步骤（就像把大象放进冰箱一样）：
- STEP1: Define a set of function
- STEP2: Goodness of the function
- STEP3: Pick the **best** function

### Techs of ML (brief intro)
![](.\images\intro_to_ml\1.png)

#### Supervised Learning
- Regression
    - output: **scalar**
    - linear regression & nonlinear regression
    - linear regression is useful in **predicting a quantitative response**
- Classification
    - output: **type**
        - Binary Classification: 'YES' or 'NO'
            e.g. SPAM filter
        - Multi Classification: class1, class2, ..., classN
            e.g. essay classification
    - linear model
    - nonlinear model
        - Deep Learning, SVM, decision tree, K-NN
        - hierarchical structure
- Structured Learning
    - 机器的输出是有结构性的
    - e.g. translation
#### Semi-supervised Learning
- target值缺失情况下使用，即你的training data中大量的target值缺失，而这些target值又无法通过自然的方式获得，只能人工标注（label），你又没有精力去告诉机器那些target值是什么（完善training set）
    - training data: input/output pair of target function
    - function output: label
#### Transfer Learning
- 使用场景：你拥有大量的labelled和unlabelled的数据，这时候你想做猫和狗的分类，但是training data里不仅仅有猫和狗的图像还有其他乱七八糟的图像
#### Unsupervised Learning
- 让机器无师自通
    - training data: all unlabelled
    - e.g. machine learns the meaning of words after reading a lot of documents
    - e.g. GAN (input images and generate new images)
#### Reinforcement Learning
- 没有正确答案，只会告诉机器一个分数，他做得好还是不好，机器要进行自我检讨来改进完善
    - e.g. 假设你是客服人员，你接到一个电话，但是说着说着对方就气得挂断电话，但你不知道为什么对方会生气，这时候你就需要反省自己哪里做错了
- 没办法做supervised learning时才用reinforcement learning



>### Reference
> - Hung-yi Lee, *Machine Learning*, Introduction of Machine Learning
> - Gareth James, *An Introduction to Statistical Learning with Applications in R*, Linear Regression

